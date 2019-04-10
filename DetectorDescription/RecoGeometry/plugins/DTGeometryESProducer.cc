// -*- C++ -*-
//
// Package:    DetectorDescription/DTGeometryESProducer
// Class:      DTGeometryESProducer
// 
/**\class DTGeometryESProducer

 Description: DT Geometry ES producer

 Implementation:
     Based on a copy of original DTGeometryESProducer
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 16 Jan 2019 10:19:37 GMT
//
//
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"
#include "CondFormats/Alignment/interface/DetectorGlobalPosition.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentErrorExtendedRcd.h"
#include "Geometry/CommonTopologies/interface/GeometryAligner.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"
#include "DetectorDescription/DDCMS/interface/MuonNumberingRcd.h"
#include "DetectorDescription/DDCMS/interface/MuonGeometryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DetectorDescription/RecoGeometry/interface/DTGeometryBuilder.h"

#include <memory>
#include <iostream>
#include <iterator>
#include <string>

using namespace edm;
using namespace std;
using namespace cms;

class DTGeometryESProducer : public ESProducer {
public:
  DTGeometryESProducer(const ParameterSet&);
  ~DTGeometryESProducer() override;

  using ReturnType = shared_ptr<DTGeometry>;
  using Detector = dd4hep::Detector;
  
  ReturnType produce(const MuonGeometryRcd& record);

private:

  using HostType = ESProductHost<DTGeometry,
				 MuonNumberingRcd,
				 DTRecoGeometryRcd>;

  void setupGeometry(MuonNumberingRcd const&, shared_ptr<HostType>&);
  void setupDBGeometry(DTRecoGeometryRcd const&, shared_ptr<HostType>&);

  ReusableObjectHolder<HostType> m_holder;

  const ESInputTag m_tag;
  const string m_alignmentsLabel;
  const string m_myLabel;
  const string m_attribute;
  const string m_value;
  bool m_applyAlignment;
  bool m_fromDDD;
};

DTGeometryESProducer::DTGeometryESProducer(const ParameterSet & iConfig)
  : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")),
    m_alignmentsLabel(iConfig.getParameter<string>("alignmentsLabel")),
    m_myLabel(iConfig.getParameter<string>("appendToDataLabel")),
    m_attribute(iConfig.getParameter<string>("attribute")),
    m_value(iConfig.getParameter<string>("value")),
    m_fromDDD(iConfig.getParameter<bool>("fromDDD"))
{
  m_applyAlignment = iConfig.getParameter<bool>("applyAlignment");

  setWhatProduced(this);

  edm::LogInfo("Geometry") << "@SUB=DTGeometryESProducer"
    << "Label '" << m_myLabel << "' "
    << (m_applyAlignment ? "looking for" : "IGNORING")
    << " alignment labels '" << m_alignmentsLabel << "'.";
}

DTGeometryESProducer::~DTGeometryESProducer(){}

std::shared_ptr<DTGeometry> 
DTGeometryESProducer::produce(const MuonGeometryRcd & record) {
  
  auto host = m_holder.makeOrGet([]() {
    return new HostType;
  });

  {
    BenchmarkGrd counter("DTGeometryESProducer");

    if(m_fromDDD) {
      host->ifRecordChanges<MuonNumberingRcd>(record,
					      [this, &host](auto const& rec) {
						setupGeometry(rec, host);
					      });
    } else {
      host->ifRecordChanges<DTRecoGeometryRcd>(record,
					       [this, &host](auto const& rec) {
						 setupDBGeometry(rec, host);
					       });
    }
  }
  //
  // Called whenever the alignments or alignment errors change
  //  
  if(m_applyAlignment) {
    // m_applyAlignment is scheduled for removal. 
    // Ideal geometry obtained by using 'fake alignment' (with m_applyAlignment = true)
    edm::ESHandle<Alignments> globalPosition;
    record.getRecord<GlobalPositionRcd>().get(m_alignmentsLabel, globalPosition);
    edm::ESHandle<Alignments> alignments;
    record.getRecord<DTAlignmentRcd>().get(m_alignmentsLabel, alignments);
    edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
    record.getRecord<DTAlignmentErrorExtendedRcd>().get(m_alignmentsLabel, alignmentErrors);
    // Only apply alignment if values exist
    if (alignments->empty() && alignmentErrors->empty() && globalPosition->empty()) {
      edm::LogInfo("Config") << "@SUB=DTGeometryRecord::produce"
        << "Alignment(Error)s and global position (label '"
        << m_alignmentsLabel << "') empty: Geometry producer (label "
        << "'" << m_myLabel << "') assumes fake and does not apply.";
    } else {
      GeometryAligner aligner;
      aligner.applyAlignments<DTGeometry>( &(*host),
                                           &(*alignments), &(*alignmentErrors),
                                           align::DetectorGlobalPosition(*globalPosition, DetId(DetId::Muon)));
    }
  }

  return host; // automatically converts to std::shared_ptr<DTGeometry>
}

void
DTGeometryESProducer::setupGeometry(const MuonNumberingRcd& record,
				    shared_ptr<HostType>& host) {
  host->clear();
  
  edm::ESHandle<MuonNumbering> mdc;
  record.get(mdc);
  
  edm::ESTransientHandle<DDDetector> cpv;
  record.getRecord<DetectorDescriptionRcd>().get(m_tag.module(), cpv);
  
  ESTransientHandle<DDSpecParRegistry> registry;
  record.getRecord<DDSpecParRegistryRcd>().get(m_tag.module(), registry);
  
  DDSpecParRefs myReg;
  {
    BenchmarkGrd b1("DTGeometryESProducer Filter Registry");
    registry->filter(myReg, m_attribute, m_value);
  }
  
  DTGeometryBuilder builder;
  builder.build(*host, &(*cpv), *mdc, myReg);
}

void
DTGeometryESProducer::setupDBGeometry( const DTRecoGeometryRcd& record,
				       std::shared_ptr<HostType>& host ) {
  // host->clear();
  
  // edm::ESHandle<RecoIdealGeometry> rig;
  // record.get(rig);
  
  // DTGeometryBuilderFromCondDB builder;
  // builder.build(host, *rig);
}

DEFINE_FWK_EVENTSETUP_MODULE(DTGeometryESProducer);

#include "DetectorDescription/DDCMS/interface/MuonGeometryRcd.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
EVENTSETUP_RECORD_REG(MuonGeometryRcd);
