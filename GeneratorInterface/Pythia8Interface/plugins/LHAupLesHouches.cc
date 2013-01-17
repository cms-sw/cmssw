#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <memory>
#include <assert.h>

#include "GeneratorInterface/Pythia8Interface/plugins/LHAupLesHouches.h"

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

  for(int i = 0; i < hepeup.NUP; i++)
    addParticle(hepeup.IDUP[i], hepeup.ISTUP[i],
                hepeup.MOTHUP[i].first, hepeup.MOTHUP[i].second,
                hepeup.ICOLUP[i].first, hepeup.ICOLUP[i].second,
                hepeup.PUP[i][0], hepeup.PUP[i][1],
                hepeup.PUP[i][2], hepeup.PUP[i][3],
                hepeup.PUP[i][4], hepeup.VTIMUP[i],
                hepeup.SPINUP[i]);

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

  //hadronisation->onBeforeHadronisation().emit();

  //event.reset();

  event->attempted();

  return true;
}
