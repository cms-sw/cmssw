/** \class HLTGetData
 *
 * See header file for documentation
 *
 *  $Date: 2007/03/28 20:33:07 $
 *  $Revision: 1.2 $
 *
 *  \author various
 *
 */

#include "HLTrigger/HLTanalyzers/interface/HLTGetData.h"

#include "DataFormats/Common/interface/Handle.h"

// system include files
#include <memory>
#include <vector>
#include <map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"

using namespace edm;
using namespace std;

//
// constructors and destructor
//
HLTGetData::HLTGetData(const edm::ParameterSet& ps)
{
  EBdigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
  EEdigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
  ESdigiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
}

HLTGetData::~HLTGetData()
{ }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTGetData::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

    edm::Handle< DetSetVector<PixelDigi> >  input;
    iEvent.getByLabel("siPixelDigis", input);
     auto_ptr<DetSetVector<PixelDigi> > NewPixelDigi(new DetSetVector<PixelDigi> );
     DetSetVector<PixelDigi>* tt = NewPixelDigi.get();
     *tt = *input.product();

     edm::Handle< edm::DetSetVector<SiStripDigi> >  input2;
     iEvent.getByLabel("siStripDigis",input2);
     auto_ptr<DetSetVector<SiStripDigi> > NewSiDigi(new DetSetVector<SiStripDigi> );
     DetSetVector<SiStripDigi>* uu = NewSiDigi.get();
     *uu = *input2.product();

     
     Handle<EBDigiCollection> EcalDigiEB;
     Handle<EEDigiCollection> EcalDigiEE;
     Handle<ESDigiCollection> EcalDigiES;
     const EBDigiCollection* EBdigis =0;
     const EEDigiCollection* EEdigis =0;
     const ESDigiCollection* ESdigis =0; 
  
  
     iEvent.getByLabel( EBdigiCollection_, EcalDigiEB );
     EBdigis = EcalDigiEB.product();
     LogDebug("DigiInfo") << "total # EBdigis: " << EBdigis->size() ;
     
     iEvent.getByLabel( EEdigiCollection_, EcalDigiEE );
     EEdigis = EcalDigiEE.product();
     LogDebug("DigiInfo") << "total # EEdigis: " << EEdigis->size() ;
    
    iEvent.getByLabel( ESdigiCollection_, EcalDigiES );
    ESdigis = EcalDigiES.product();
    LogDebug("DigiInfo") << "total # ESdigis: " << ESdigis->size() ;
   



}
