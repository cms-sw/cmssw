#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <memory>
#include <assert.h>

#include "GeneratorInterface/Pythia8Interface/plugins/LHAupLesHouches.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace Pythia8;


bool LHAupLesHouches::setInit()
{
  if (!runInfo) return false;
  const lhef::HEPRUP &heprup = *runInfo->getHEPRUP();

  setBeamA(heprup.IDBMUP.first, heprup.EBMUP.first,
           heprup.PDFGUP.first, heprup.PDFSUP.first);
  setBeamB(heprup.IDBMUP.second, heprup.EBMUP.second,
           heprup.PDFGUP.second, heprup.PDFSUP.second);
  setStrategy(heprup.IDWTUP);

  for(int i = 0; i < heprup.NPRUP; i++)
    addProcess(heprup.LPRUP[i], heprup.XSECUP[i],
	       heprup.XERRUP[i], heprup.XMAXUP[i]);

  //hadronisation->onInit().emit();

  //runInfo.reset();
    
  //fill SLHA header information if available
  std::vector<std::string> slha = runInfo->findHeader("slha");
  if (!slha.empty()) {
    std::string slhaheader;
    for(std::vector<std::string>::const_iterator iter = slha.begin(); iter != slha.end(); ++iter) {
      slhaheader.append(*iter);
    }
    infoPtr->setHeader("slha",slhaheader);
  }  
  
  //work around missing initialization inside pythia8
  infoPtr->eventAttributes = new std::map<std::string, std::string >;
  
  
  return true;
}


bool LHAupLesHouches::setEvent(int inProcId)
{
  if (!event) return false;
	
  if ( event->getReadAttempts() > 0 ) return false; // record already tried
	
  const lhef::HEPEUP &hepeup = *event->getHEPEUP();
	
  if ( !hepeup.NUP ) return false;	

  setProcess(hepeup.IDPRUP, hepeup.XWGTUP, hepeup.SCALUP,
             hepeup.AQEDUP, hepeup.AQCDUP);

  const std::vector<float> &scales = event->scales();
    
  unsigned int iscale = 0;
  for(int i = 0; i < hepeup.NUP; i++) {
    //retrieve scale corresponding to each particle
    double scalein = -1.;
    
    //handle clustering scales if present,
    //applies to outgoing partons only
    if (setScalesFromLHEF_ && scales.size()>0 && hepeup.ISTUP[i]==1) {
      if (iscale>=scales.size()) {
        edm::LogError("InvalidLHEInput") << "Pythia8 requires"
                                    << "cluster scales for all outgoing partons or for none" 
                                    << std::endl;
      }
      scalein = scales[iscale];
      ++iscale;
    }
        
    addParticle(hepeup.IDUP[i], hepeup.ISTUP[i],
                hepeup.MOTHUP[i].first, hepeup.MOTHUP[i].second,
                hepeup.ICOLUP[i].first, hepeup.ICOLUP[i].second,
                hepeup.PUP[i][0], hepeup.PUP[i][1],
                hepeup.PUP[i][2], hepeup.PUP[i][3],
                hepeup.PUP[i][4], hepeup.VTIMUP[i],
                hepeup.SPINUP[i],scalein);
  }
  
  infoPtr->eventAttributes->clear();
  
  //fill parton multiplicities if available
  int npLO = event->npLO();
  int npNLO = event->npNLO();

  //default value of -99 indicates tags were not present in the original LHE file
  //don't pass to pythia in this case to emulate pythia internal lhe reader behaviour
  if (npLO!=-99) {
    char buffer [100];
    snprintf( buffer, 100, "%i",npLO);    
    (*infoPtr->eventAttributes)["npLO"] = buffer;
  }
  if (npNLO!=-99) {
    char buffer [100];
    snprintf( buffer, 100, "%i",npNLO);    
    (*infoPtr->eventAttributes)["npNLO"] = buffer;
  }
  
  const lhef::LHEEvent::PDF *pdf = event->getPDF();
  if (pdf) {
    this->setPdf(pdf->id.first, pdf->id.second,
                 pdf->x.first, pdf->x.second,
                 pdf->scalePDF,
                 pdf->xPDF.first, pdf->xPDF.second, true);
  }
  else {
    this->setPdf(hepeup.IDUP[0], hepeup.IDUP[1],
                 hepeup.PUP[0][3] / runInfo->getHEPRUP()->EBMUP.first,
                 hepeup.PUP[1][3] / runInfo->getHEPRUP()->EBMUP.second,
                 0., 0., 0., false);
  }

  event->attempted();

  return true;
}
