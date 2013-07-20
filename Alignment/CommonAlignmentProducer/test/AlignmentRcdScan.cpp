// \file AlignmentRcdScan.cpp
//
// \author    : Andreas Mussgiller
// Revision   : $Revision: 1.2 $
// last update: $Date: 2012/03/28 15:44:19 $
// by         : $Author: flucke $

#include <string>
#include <map>
#include <vector>

#include <TMath.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/DTAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/CSCAlignmentRcd.h"

#include "FWCore/Framework/interface/ESWatcher.h"

class AlignmentRcdScan : public edm::EDAnalyzer
{
public:

  enum Mode {
    Unknown=0,
    Tk=1,
    DT=2,
    CSC=3
  };

  explicit AlignmentRcdScan( const edm::ParameterSet& iConfig );
  ~AlignmentRcdScan();

  virtual void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup); 

private:

  void inspectRecord(const std::string & rcdname,
		     const edm::Event& evt, 
		     const edm::ESHandle<Alignments> & alignments);

  int mode_;
  bool verbose_;

  edm::ESWatcher<TrackerAlignmentRcd> watchTk_;
  edm::ESWatcher<DTAlignmentRcd>      watchDT_;
  edm::ESWatcher<CSCAlignmentRcd>     watchCSC_;

  Alignments *refAlignments_;
};

AlignmentRcdScan::AlignmentRcdScan( const edm::ParameterSet& iConfig )
  :verbose_(iConfig.getUntrackedParameter<bool>("verbose")),
   refAlignments_(0)
{
  std::string modestring = iConfig.getUntrackedParameter<std::string>("mode");
  if (modestring=="Tk") {
    mode_ = Tk;
  } else if (modestring=="DT") {
    mode_ = DT;
  } else if (modestring=="CSC") {
    mode_ = CSC;
  } else {
    mode_ = Unknown;
  }

  if (mode_==Unknown) {
    throw cms::Exception("BadConfig") << "Mode " << modestring << " not known";
  }
}

AlignmentRcdScan::~AlignmentRcdScan()
{
  if (refAlignments_) delete refAlignments_;
}

void AlignmentRcdScan::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  if (mode_==Tk && watchTk_.check(evtSetup)) {
    edm::ESHandle<Alignments> alignments;
    evtSetup.get<TrackerAlignmentRcd>().get(alignments);
    inspectRecord("TrackerAlignmentRcd", evt, alignments);
  }
  if (mode_==DT && watchDT_.check(evtSetup)) {
    edm::ESHandle<Alignments> alignments;
    evtSetup.get<DTAlignmentRcd>().get(alignments);  
    inspectRecord("DTAlignmentRcd", evt, alignments);
  }
  if (mode_==CSC && watchCSC_.check(evtSetup)) {
    edm::ESHandle<Alignments> alignments;
    evtSetup.get<CSCAlignmentRcd>().get(alignments);    
    inspectRecord("CSCAlignmentRcd", evt, alignments);
  }
}
 
void AlignmentRcdScan::inspectRecord(const std::string & rcdname,
				     const edm::Event& evt, 
				     const edm::ESHandle<Alignments> & alignments)
{
  std::cout << rcdname << " content starting from run " << evt.run();
  
  if (verbose_==false) {
    std::cout << std::endl;
    return;
  }

  std::cout << " with " << alignments->m_align.size() << " entries" << std::endl;
  
  if (refAlignments_) {

    std::cout << "  Compared to previous record:" << std::endl;
    
    double meanX = 0;
    double rmsX = 0;
    double meanY = 0;
    double rmsY = 0;
    double meanZ = 0;
    double rmsZ = 0;
    double meanR = 0;
    double rmsR = 0;
    double dPhi;
    double meanPhi = 0;
    double rmsPhi = 0;
    
    std::vector<AlignTransform>::const_iterator iref = refAlignments_->m_align.begin();
    for (std::vector<AlignTransform>::const_iterator i = alignments->m_align.begin();
	 i != alignments->m_align.end();
	 ++i, ++iref) {
      
      meanX += i->translation().x() - iref->translation().x();
      rmsX += pow(i->translation().x() - iref->translation().x(), 2);
 
      meanY += i->translation().y() - iref->translation().y();
      rmsY += pow(i->translation().y() - iref->translation().y(), 2);
      
      meanZ += i->translation().z() - iref->translation().z();
      rmsZ += pow(i->translation().z() - iref->translation().z(), 2);
      
      meanR += i->translation().perp() - iref->translation().perp();
      rmsR += pow(i->translation().perp() - iref->translation().perp(), 2);

      dPhi = i->translation().phi() - iref->translation().phi();
      if (dPhi>TMath::Pi()) dPhi -= 2.0*TMath::Pi();
      if (dPhi<-TMath::Pi()) dPhi += 2.0*TMath::Pi();

      meanPhi += dPhi;
      rmsPhi += dPhi*dPhi;
    }

    meanX /= alignments->m_align.size();
    rmsX /= alignments->m_align.size();
    meanY /= alignments->m_align.size();
    rmsY /= alignments->m_align.size();
    meanZ /= alignments->m_align.size();
    rmsZ /= alignments->m_align.size();
    meanR /= alignments->m_align.size();
    rmsR /= alignments->m_align.size();
    meanPhi /= alignments->m_align.size();
    rmsPhi /= alignments->m_align.size();
    
    std::cout << "    mean X shift:   " 
	      << std::setw(12) << std::scientific << std::setprecision(3) << meanX
	      << " (RMS = " << sqrt(rmsX) << ")" << std::endl;
    std::cout << "    mean Y shift:   " 
	      << std::setw(12) << std::scientific << std::setprecision(3) << meanY
	      << " (RMS = " << sqrt(rmsY) << ")" << std::endl;
    std::cout << "    mean Z shift:   " 
	      << std::setw(12) << std::scientific << std::setprecision(3) << meanZ
	      << " (RMS = " << sqrt(rmsZ) << ")" << std::endl;
    std::cout << "    mean R shift:   " 
	      << std::setw(12) << std::scientific << std::setprecision(3) << meanR
	      << " (RMS = " << sqrt(rmsR) << ")" << std::endl;
    std::cout << "    mean Phi shift: " 
	      << std::setw(12) << std::scientific << std::setprecision(3) << meanPhi
	      << " (RMS = " << sqrt(rmsPhi) << ")" << std::endl;
    
    delete refAlignments_;
  }

  refAlignments_ = new Alignments(*alignments);

  std::cout << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(AlignmentRcdScan);
