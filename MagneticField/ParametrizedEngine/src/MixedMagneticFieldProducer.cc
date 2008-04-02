/** \file
 *
 *  $Date: 2008/03/28 16:49:25 $
 *  $Revision: 1.2 $
 *  \author Massimiliano Chiorboli, updated NA 03/08
 */

#include "MagneticField/ParametrizedEngine/src/MixedMagneticFieldProducer.h"
#include "MagneticField/ParametrizedEngine/src/MixedMagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <string>
#include <iostream>

using namespace std;
using namespace edm;
using namespace magneticfield;


MixedMagneticFieldProducer::MixedMagneticFieldProducer(const edm::ParameterSet& iConfig) : pset(iConfig) {
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}


MixedMagneticFieldProducer::~MixedMagneticFieldProducer()
{
}


std::auto_ptr<MagneticField>
MixedMagneticFieldProducer::produce(const IdealMagneticFieldRecord& iRecord)
{
  string parLabel = pset.getUntrackedParameter<string>("parametrizationLabel");
  string fmLabel = pset.getUntrackedParameter<string>("fullMapLabel");
  double scale = pset.getParameter<double>("fullMapScale");
  
  if (fabs(scale-1.) > 0.001) {
    edm::LogWarning("MagneticField|MixedMagneticFieldProducer") << " You are using MixedMagneticField with fullMapScale = " << scale << ". The resulting magnetic field is UNPHYSICAL.";
  }

  // Get slave fields
  edm::ESHandle<MagneticField> paramField;
  iRecord.get(parLabel,paramField);
  
  edm::ESHandle<MagneticField> fullField;
  iRecord.get(fmLabel,fullField);

  std::auto_ptr<MagneticField> result(new MixedMagneticField(paramField.product(), fullField.product(), scale));
  
  return result;
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(MixedMagneticFieldProducer);
