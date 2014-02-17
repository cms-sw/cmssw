// -*- C++ -*-
//
// Package:    CSCDigiValidator
// Class:      CSCDigiValidator
// 
/**\class CSCDigiValidator CSCDigiValidator.cc UserCode/CSCDigiValidator/src/CSCDigiValidator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lindsey Gray
//         Created:  Tue Jul 28 18:04:11 CEST 2009
// $Id: CSCDigiValidator.cc,v 1.3 2011/11/01 16:31:54 asakharo Exp $
//
//


// system include files
#include <memory>
#include <map>
#include <vector>
#include <algorithm>

// user include files
#include "EventFilter/CSCRawToDigi/interface/CSCDigiValidator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"




//
// constructors and destructor
//
CSCDigiValidator::CSCDigiValidator(const edm::ParameterSet& iConfig) :
  wire1(iConfig.getUntrackedParameter<edm::InputTag>("inputWire")),
  strip1(iConfig.getUntrackedParameter<edm::InputTag>("inputStrip")),
  comp1(iConfig.getUntrackedParameter<edm::InputTag>("inputComp")),
  clct1(iConfig.getUntrackedParameter<edm::InputTag>("inputCLCT")),
  alct1(iConfig.getUntrackedParameter<edm::InputTag>("inputALCT")),
  lct1(iConfig.getUntrackedParameter<edm::InputTag>("inputCorrLCT")),
  csctf1(iConfig.getUntrackedParameter<edm::InputTag>("inputCSCTF")),
  csctfstubs1(iConfig.getUntrackedParameter<edm::InputTag>("inputCSCTFStubs")),
  wire2(iConfig.getUntrackedParameter<edm::InputTag>("repackWire")),
  strip2(iConfig.getUntrackedParameter<edm::InputTag>("repackStrip")),
  comp2(iConfig.getUntrackedParameter<edm::InputTag>("repackComp")),
  clct2(iConfig.getUntrackedParameter<edm::InputTag>("repackCLCT")),
  alct2(iConfig.getUntrackedParameter<edm::InputTag>("repackALCT")),
  lct2(iConfig.getUntrackedParameter<edm::InputTag>("repackCorrLCT")),
  csctf2(iConfig.getUntrackedParameter<edm::InputTag>("repackCSCTF")),
  csctfstubs2(iConfig.getUntrackedParameter<edm::InputTag>("repackCSCTFStubs"))
  //  reorderStrips(iConfig.getUntrackedParameter<bool>("applyStripReordering",true))
{
   //now do what ever initialization is needed
  //  produces<std::map<std::string,unsigned> >(); // # of errors enumerated by error type
}


CSCDigiValidator::~CSCDigiValidator()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
bool
CSCDigiValidator::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  bool _err = false;
  using namespace edm;

  typedef std::map<CSCDetId,
    std::pair<std::vector<CSCWireDigi>,std::vector<CSCWireDigi> > > 
    matchingDetWireCollection;
  typedef std::map<CSCDetId,
    std::pair<std::vector<CSCStripDigi>,std::vector<CSCStripDigi> > > 
    matchingDetStripCollection;
  typedef std::map<CSCDetId,
    std::pair<std::vector<CSCComparatorDigi>,std::vector<CSCComparatorDigi> > > 
    matchingDetComparatorCollection;
  typedef std::map<CSCDetId,
    std::pair<std::vector<CSCCLCTDigi>,std::vector<CSCCLCTDigi> > > 
    matchingDetCLCTCollection;
  typedef std::map<CSCDetId,
    std::pair<std::vector<CSCALCTDigi>,std::vector<CSCALCTDigi> > > 
    matchingDetALCTCollection;
  typedef std::map<CSCDetId,
    std::pair<std::vector<CSCCorrelatedLCTDigi>,std::vector<CSCCorrelatedLCTDigi> > > 
    matchingDetLCTCollection;

  // std::auto_ptr<std::map<std::string,unsigned> >
  //    errors(new std::map<std::string,unsigned>);

  edm::ESHandle<CSCChamberMap> hcham;
  iSetup.get<CSCChamberMapRcd>().get(hcham); 
  const CSCChamberMap* theMapping = hcham.product();
    
  Handle<CSCWireDigiCollection> _wi,_swi;
  Handle<CSCStripDigiCollection> _st,_sst;
  Handle<CSCComparatorDigiCollection> _cmp,_scmp;
  Handle<CSCCLCTDigiCollection> _clct, _sclct;
  Handle<CSCALCTDigiCollection> _alct, _salct;
  Handle<CSCCorrelatedLCTDigiCollection> _lct, _slct;
  Handle<L1CSCTrackCollection> _trk, _strk;
  Handle<CSCTriggerContainer<csctf::TrackStub> > _dt, _sdt;

   // get wire digis before and after unpacking
   iEvent.getByLabel(wire1,_swi);
   iEvent.getByLabel(wire2,_wi);
   
   //get strip digis before and after unpacking
   iEvent.getByLabel(strip1,_sst);
   iEvent.getByLabel(strip2,_st);

   //get comparator digis before and after unpacking
   iEvent.getByLabel(comp1,_scmp);
   iEvent.getByLabel(comp2,_cmp);

   //get clcts
   iEvent.getByLabel(clct1,_sclct);
   iEvent.getByLabel(clct2,_clct);

   //get alcts
   iEvent.getByLabel(alct1,_salct);
   iEvent.getByLabel(alct2,_alct);

   //get lcts
   iEvent.getByLabel(lct1,_slct);
   iEvent.getByLabel(lct2,_lct);

   //get l1 tracks
   iEvent.getByLabel(csctf1,_strk);
   iEvent.getByLabel(csctfstubs1,_sdt);
   iEvent.getByLabel(csctf2,_trk);
   iEvent.getByLabel(csctfstubs2,_dt);
   
   //get DT stubs for L1 Tracks

   CSCWireDigiCollection::DigiRangeIterator 
     wi = _wi->begin(), swi= _swi->begin();
   CSCStripDigiCollection::DigiRangeIterator 
     st = _st->begin(), sst = _sst->begin();
   CSCComparatorDigiCollection::DigiRangeIterator 
     cmp = _cmp->begin(), scmp = _scmp->begin();
   CSCCLCTDigiCollection::DigiRangeIterator
     clct = _clct->begin(), sclct = _sclct->begin();
   CSCALCTDigiCollection::DigiRangeIterator
     alct = _alct->begin(), salct = _salct->begin();
   CSCCorrelatedLCTDigiCollection::DigiRangeIterator
     lct = _lct->begin(), slct = _slct->begin();
   L1CSCTrackCollection::const_iterator 
     trk = _trk->begin(), strk = _strk->begin();
   std::vector<csctf::TrackStub>::const_iterator
     dt = _dt->get().begin(), sdt = _sdt->get().begin();
     // WARNING 5_0_X
     dt++; dt--; sdt++; sdt--;

   //per detID, create lists of various digi types
   matchingDetWireCollection wires;
   matchingDetStripCollection strips;
   matchingDetComparatorCollection comps;
   matchingDetCLCTCollection clcts;
   matchingDetALCTCollection alcts;
   matchingDetLCTCollection lcts,trackstubs;
   CSCTriggerContainer<csc::L1Track> tracks, simtracks;
   
   //wires
   for(;wi != _wi->end();++wi)
     {
       CSCWireDigiCollection::const_iterator 
	 b=(*wi).second.first,e=(*wi).second.second;
       std::vector<CSCWireDigi>::iterator 
	 beg=wires[(*wi).first].first.end();
       wires[(*wi).first].first.insert(beg,b,e);
     }
   for(;swi != _swi->end();++swi)
     {
       CSCWireDigiCollection::const_iterator 
	 b=(*swi).second.first,e=(*swi).second.second;
       //convert sim ring 4(ME1/a) to ring 1
       CSCDetId _id = (*swi).first;
       if((*swi).first.ring() == 4)
	 _id = CSCDetId((*swi).first.endcap(),(*swi).first.station(),
			1, (*swi).first.chamber(),(*swi).first.layer());

       std::vector<CSCWireDigi>::iterator 
	 beg=wires[_id].second.end();
       
       wires[_id].second.insert(beg,b,e);
       // automatically combine wire digis after each insertion
       wires[_id].second = sanitizeWireDigis(wires[_id].second.begin(),
					     wires[_id].second.end());
     }
   
   //strips
   for(;st != _st->end();++st)
     {
       CSCStripDigiCollection::const_iterator 
	 b=(*st).second.first,e=(*st).second.second;
       std::vector<CSCStripDigi>::iterator 
	 beg=strips[(*st).first].first.end();
              
       //need to remove strips with no active ADCs
       std::vector<CSCStripDigi> zs = zeroSupStripDigis(b,e);

       strips[(*st).first].first.insert(beg,zs.begin(),zs.end());
     }
   for(;sst != _sst->end();++sst)
     {
       CSCStripDigiCollection::const_iterator 
	 b=(*sst).second.first,e=(*sst).second.second;
       // conversion of ring 4->1 not necessary here
       CSCDetId _id = (*sst).first;
       //if((*sst).first.ring() == 4)	 
       //	 _id = CSCDetId((*sst).first.endcap(),(*sst).first.station(),
       //		1, (*sst).first.chamber(),(*sst).first.layer());

       std::vector<CSCStripDigi>::iterator 
	 beg=strips[_id].second.end();       
       
       std::vector<CSCStripDigi> relab = relabelStripDigis(theMapping,(*sst).first,b,e);
        
       strips[_id].second.insert(beg,relab.begin(),relab.end());              
       //strips[_id].second.insert(beg,b,e);  
     }
   
   //comparators
   for(;cmp != _cmp->end();++cmp)
     {
       CSCComparatorDigiCollection::const_iterator 
	 b=(*cmp).second.first,e=(*cmp).second.second;
       std::vector<CSCComparatorDigi>::iterator 
	 beg=comps[(*cmp).first].first.end();

       comps[(*cmp).first].first.insert(beg,b,e);
     }
   for(;scmp != _scmp->end();++scmp)
     {
       CSCComparatorDigiCollection::const_iterator 
	 b=(*scmp).second.first,e=(*scmp).second.second;
       // convert sim ring 4 (ME1/a) to ring 1
       CSCDetId _id = (*scmp).first;
       if((*scmp).first.ring() == 4)
         _id = CSCDetId((*scmp).first.endcap(),(*scmp).first.station(),
			1, (*scmp).first.chamber(),(*scmp).first.layer());
       
       std::vector<CSCComparatorDigi>::iterator 
	 beg=comps[_id].second.begin();

       if((*scmp).first.ring()==4)
	 beg=comps[_id].second.end();

       std::vector<CSCComparatorDigi> zs = 
	 zeroSupCompDigis(b,e);			  

       std::vector<CSCComparatorDigi> relab = 
	 relabelCompDigis(theMapping,(*scmp).first,
			  zs.begin(),
			  zs.end());

       comps[_id].second.insert(beg,relab.begin(),relab.end());
     }

   //CLCTs
   for(;clct != _clct->end();++clct)
     {
       CSCCLCTDigiCollection::const_iterator 
	 b=(*clct).second.first,e=(*clct).second.second;
       std::vector<CSCCLCTDigi>::iterator 
	 beg=clcts[(*clct).first].first.end();

       clcts[(*clct).first].first.insert(beg,b,e);
     }
   for(;sclct != _sclct->end();++sclct)
     {
       CSCCLCTDigiCollection::const_iterator 
	 b=(*sclct).second.first,e=(*sclct).second.second;
       // convert sim ring 4 (ME1/a) to ring 1
       CSCDetId _id = (*sclct).first;
       if((*sclct).first.ring() == 4)
         _id = CSCDetId((*sclct).first.endcap(),(*sclct).first.station(),
			1, (*sclct).first.chamber(),(*sclct).first.layer());
       
       std::vector<CSCCLCTDigi>::iterator 
	 beg=clcts[_id].second.begin();

       if((*sclct).first.ring()==4)
	 beg=clcts[_id].second.end();
       
       clcts[_id].second.insert(beg,b,e);
     }

   //ALCTs
   for(;alct != _alct->end();++alct)
     {
       CSCALCTDigiCollection::const_iterator 
	 b=(*alct).second.first,e=(*alct).second.second;
       std::vector<CSCALCTDigi>::iterator 
	 beg=alcts[(*alct).first].first.end();

       alcts[(*alct).first].first.insert(beg,b,e);
     }
   for(;salct != _salct->end();++salct)
     {
       CSCALCTDigiCollection::const_iterator 
	 b=(*salct).second.first,e=(*salct).second.second;
       // convert sim ring 4 (ME1/a) to ring 1
       CSCDetId _id = (*salct).first;
       if((*salct).first.ring() == 4)
         _id = CSCDetId((*salct).first.endcap(),(*salct).first.station(),
			1, (*salct).first.chamber(),(*salct).first.layer());
       
       std::vector<CSCALCTDigi>::iterator 
	 beg=alcts[_id].second.begin();

       if((*salct).first.ring()==4)
	 beg=alcts[_id].second.end();
       
       alcts[_id].second.insert(beg,b,e);
     }

   // Correlated LCTs
   for(;lct != _lct->end();++lct)
     {
       CSCCorrelatedLCTDigiCollection::const_iterator 
	 b=(*lct).second.first,e=(*lct).second.second;
       std::vector<CSCCorrelatedLCTDigi>::iterator 
	 beg=lcts[(*lct).first].first.end();

       lcts[(*lct).first].first.insert(beg,b,e);
     }
   for(;slct != _slct->end();++slct)
     {
       CSCCorrelatedLCTDigiCollection::const_iterator 
	 b=(*slct).second.first,e=(*slct).second.second;
       // convert sim ring 4 (ME1/a) to ring 1
       CSCDetId _id = (*slct).first;
       if((*slct).first.ring() == 4)
         _id = CSCDetId((*slct).first.endcap(),(*slct).first.station(),
			1, (*slct).first.chamber(),(*slct).first.layer());
       
       std::vector<CSCCorrelatedLCTDigi>::iterator 
	 beg=lcts[_id].second.begin();

       if((*slct).first.ring()==4)
	 beg=lcts[_id].second.end();
       
       lcts[_id].second.insert(beg,b,e);
     }
   // remove attached LCT digis from tracks, should be put into their own collection and checked separately
   for(; trk != _trk->end(); ++trk)
     {
       tracks.push_back(trk->first);
       
     }
   for(;strk != _strk->end(); ++strk)
     {
       simtracks.push_back(strk->first);
     }

   //now loop through each set and process if there are differences!
   matchingDetWireCollection::const_iterator w;
   matchingDetStripCollection::const_iterator s;
   matchingDetComparatorCollection::const_iterator c;
   matchingDetCLCTCollection::const_iterator cl;
   matchingDetALCTCollection::const_iterator al;
   matchingDetLCTCollection::const_iterator lc;
   
   for(w = wires.begin(); w != wires.end(); ++w)
     {
       if(w->second.first.size() != w->second.second.size())
	 {
	   std::cout << "Major error! # of wire digis in detID: " << w->first 
		     << " is not equal between sim and unpacked!" << std::endl;
	   //eventually do more in this case!

	   std::vector<CSCWireDigi> a = w->second.second;
	   std::vector<CSCWireDigi> b = w->second.first;
	   std::cout << "SIM OUTPUT:" << std::endl;
	   for(std::vector<CSCWireDigi>::const_iterator i = a.begin(); i != a.end(); ++i)
	     i->print();
	   std::cout << "UNPACKER OUTPUT:" << std::endl;
	   for(std::vector<CSCWireDigi>::const_iterator i = b.begin(); i != b.end(); ++i)
	     i->print();
	     
	 }
       int max = std::min(w->second.first.size(),w->second.second.size());
       std::vector<CSCWireDigi> cv = w->second.first;
       std::vector<CSCWireDigi> sv = w->second.second;
       for(int i = 0; i < max; ++i)
	 {
	   if(sv[i].getWireGroup() != cv[i].getWireGroup())
	     {
	       std::cout << "In detId: " << w->first << std::endl;
	       std::cout << "Wire Groups do not match: " << sv[i].getWireGroup() 
			 << " != " << cv[i].getWireGroup() << std::endl;	       
	     }
	   if(sv[i].getTimeBin() != cv[i].getTimeBin())
	     {
	       std::cout << "In detId: " << w->first << std::endl;
	       std::cout << "First Time Bins do not match: " << sv[i].getTimeBin()
			 << " != " << cv[i].getTimeBin() << std::endl;	       
	     }	   
	   if(sv[i].getTimeBinWord() != cv[i].getTimeBinWord())
	     {
	       std::cout << "In detId: " << w->first << std::endl;
	       std::cout << "Time Bin Words do not match: " << sv[i].getTimeBinWord()
			 << " != " << cv[i].getTimeBinWord() << std::endl;
	     }	   
	 }
     }
   for(s = strips.begin(); s != strips.end(); ++s)
     {
       if(s->second.first.size() != s->second.second.size())
	 {
	   std::cout << "Major error! # of strip digis in detID: " << s->first 
		     << " is not equal between sim and unpacked!" << std::endl;
	   //eventually do more in this case!

	   std::vector<CSCStripDigi> a = s->second.second;
	   std::vector<CSCStripDigi> b = s->second.first;
	   std::cout << "SIM OUTPUT:" << std::endl;
	   for(std::vector<CSCStripDigi>::const_iterator i = a.begin(); i != a.end(); ++i)
	     i->print();
	   std::cout << "UNPACKER OUTPUT:" << std::endl;
	   for(std::vector<CSCStripDigi>::const_iterator i = b.begin(); i != b.end(); ++i)
	     i->print();	     
	 }
       int max = std::min(s->second.first.size(),s->second.second.size());
       std::vector<CSCStripDigi> cv = s->second.first;
       std::vector<CSCStripDigi> sv = s->second.second;
       for(int i = 0; i < max; ++i)
	 {
	   bool me1a = s->first.station()==1 && s->first.ring()==4;
	   bool me1b = s->first.station()==1 && s->first.ring()==1;
	   bool zplus = s->first.endcap()==1;
	   int k=i;
	   
	   if(me1a && zplus) k=max-i-1;
	   if(me1b && !zplus) k=max-i-1;

	   if(sv[k].getStrip() != cv[i].getStrip())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "Strips do not match: " << sv[k].getStrip() 
			 << " != " << cv[i].getStrip() << std::endl;	       
	     }
	   if(sv[k].getADCCounts().size() != cv[i].getADCCounts().size())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "ADC Readouts not of equal size!" << std::endl;
	       std::cout << sv[k].getADCCounts().size() << ' ' 
			 << cv[i].getADCCounts().size() << std::endl;
	     }
	   else
	     {
	       std::vector<int> sADC = sv[k].getADCCounts();
	       std::vector<int> uADC = cv[i].getADCCounts();

	       for(unsigned iadc = 0; iadc < sADC.size(); ++iadc)
		 if(sADC[iadc] != uADC[iadc])
		   {
		     std::cout << "In detId: " << s->first << std::endl;
		     std::cout << "ADC counts not equal at index: " << iadc << std::endl
			       << std::hex <<sADC[iadc] << " != " << uADC[iadc] << std::dec
			       << std::endl;	
		   }	 
	     }
	   if(sv[k].getADCOverflow().size() != cv[i].getADCOverflow().size())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "ADC Overflows not of equal size!" << std::endl;
	       std::cout << sv[k].getADCOverflow().size() << ' ' 
			 << cv[i].getADCOverflow().size() << std::endl;
	     }
	   else
	     {
	       std::vector<uint16_t> sADC = sv[k].getADCOverflow();
	       std::vector<uint16_t> uADC = cv[i].getADCOverflow();

	       for(unsigned iadc = 0; iadc < sADC.size(); ++iadc)
		 if(sADC[iadc] != uADC[iadc])
		   {
		     std::cout << "In detId: " << s->first << std::endl;
		     std::cout << "ADC overflows not equal at index: " << iadc << std::endl
			       << std::hex <<sADC[iadc] << " != " << uADC[iadc] << std::dec
			       << std::endl;		 
		   }
	     }
	   if(sv[k].getOverlappedSample().size() != cv[i].getOverlappedSample().size())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "Overlapped Samples not of equal size!" << std::endl;
	       std::cout << sv[k].getOverlappedSample().size() << ' ' 
			 << cv[i].getOverlappedSample().size() << std::endl;
	     }
	   else
	     {
	       std::vector<uint16_t> sADC = sv[k].getOverlappedSample();
	       std::vector<uint16_t> uADC = cv[i].getOverlappedSample();

	       for(unsigned iadc = 0; iadc < sADC.size(); ++iadc)
		 if(sADC[iadc] != uADC[iadc])
		   {
		     std::cout << "In detId: " << s->first << std::endl;
		     std::cout << "Overlapped Samples not equal at index: " << iadc << std::endl
			       << std::hex <<sADC[iadc] << " != " << uADC[iadc] << std::dec
			       << std::endl;		 
		   }
	     }
	   if(sv[k].getErrorstat().size() != cv[i].getErrorstat().size())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "Errorstat not of equal size!" << std::endl;
	       std::cout << sv[k].getErrorstat().size() << ' ' 
			 << cv[i].getErrorstat().size() << std::endl;
	     }
	   else
	     {
	       std::vector<uint16_t> sADC = sv[k].getErrorstat();
	       std::vector<uint16_t> uADC = cv[i].getErrorstat();

	       for(unsigned iadc = 0; iadc < sADC.size(); ++iadc)
		 if(sADC[iadc] != uADC[iadc])
		   {
		     std::cout << "In detId: " << s->first << std::endl;
		     std::cout << "Errorstat not equal at index: " << iadc << std::endl
			       << std::hex <<sADC[iadc] << " != " << uADC[iadc] << std::dec
			       << std::endl;		 
		   }
	     }
	   if(sv[k].pedestal() != cv[i].pedestal())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "Pedestals not equal: " << sv[k].pedestal() << " != " 
			 << cv[i].pedestal() << std::endl;
	     }
	   if(sv[k].amplitude() != cv[i].amplitude())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "Amplitudes not equal: " << sv[k].amplitude() << " != " 
			 << cv[i].amplitude() << std::endl;
	     }
	 }
     }
   for(c = comps.begin(); c != comps.end(); ++c)
     {
       if(c->second.first.size() != c->second.second.size())
	 {
	   std::cout << "Major error! # of comparator digis in detID: " << c->first
		     << " is not equal between sim and unpacked!" << std::endl;
	   //eventually do more in this case!

	   std::vector<CSCComparatorDigi> a = c->second.second;
	   std::vector<CSCComparatorDigi> b = c->second.first;
	   std::cout << "SIM OUTPUT:" << std::endl;
	   for(std::vector<CSCComparatorDigi>::const_iterator i = a.begin(); i != a.end(); ++i)
	     i->print();
	   std::cout << "UNPACKER OUTPUT:" << std::endl;
	   for(std::vector<CSCComparatorDigi>::const_iterator i = b.begin(); i != b.end(); ++i)
	     i->print();	     
	 }
       int max = std::min(c->second.first.size(),c->second.second.size());
       std::vector<CSCComparatorDigi> cv = c->second.first;
       std::vector<CSCComparatorDigi> sv = c->second.second;
       for(int i = 0; i < max; ++i)
	 {	   
	   if(sv[i].getStrip() != cv[i].getStrip())
	     {
	       std::cout << "In detId: " << s->first << std::endl;
	       std::cout << "Comparator strips do not match: " << sv[i].getStrip() 
			 << " != " << cv[i].getStrip() << std::endl;
	     }
	   if(sv[i].getComparator() != cv[i].getComparator())
	     {	       
	       std::cout << "In detId: " << c->first << std::endl;
	       std::cout << "Comparators do not match: " << sv[i].getComparator()
			 << " != " << cv[i].getComparator() << std::endl;	      
	     }
	   if(sv[i].getTimeBinWord() != cv[i].getTimeBinWord())
	     {
	       std::cout << "In detId: " << c->first << std::endl;
	       std::cout << "Comparator time bins words do not match: " << sv[i].getTimeBinWord()
			 << " != " << cv[i].getTimeBinWord() << std::endl;
	     }
	 }
     }
   for(cl = clcts.begin(); cl != clcts.end(); ++cl)
     {
       if(cl->second.first.size() != cl->second.second.size())
	 {
	   std::cout << "Major error! # of CLCT digis in detID: " << cl->first
		     << " is not equal between sim and unpacked!" << std::endl;
	   //eventually do more in this case!

	   std::vector<CSCCLCTDigi> a = cl->second.second;
	   std::vector<CSCCLCTDigi> b = cl->second.first;
	   std::cout << "SIM OUTPUT:" << std::endl;
	   for(std::vector<CSCCLCTDigi>::const_iterator i = a.begin(); i != a.end(); ++i)
	     i->print();
	   std::cout << "UNPACKER OUTPUT:" << std::endl;
	   for(std::vector<CSCCLCTDigi>::const_iterator i = b.begin(); i != b.end(); ++i)
	     i->print();	     
	 }
       int max = std::min(cl->second.first.size(),cl->second.second.size());
       std::vector<CSCCLCTDigi> cv = cl->second.first;
       std::vector<CSCCLCTDigi> sv = cl->second.second;
       for(int i = 0; i < max; ++i)
	 {
	   if(cv[i].getKeyStrip() != sv[i].getKeyStrip())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT key strips do not match: " << sv[i].getKeyStrip()
			 << " != " << cv[i].getKeyStrip() << std::endl;
	     }
	   if(cv[i].getStrip() != sv[i].getStrip())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT strips do not match: " << sv[i].getStrip()
			 << " != " << cv[i].getStrip() << std::endl;
	     }	   
	   if(cv[i].isValid() != sv[i].isValid())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT Valid bits do not match: " << sv[i].isValid() 
			 << " != " << cv[i].isValid() << std::endl;
	     }
	   if(cv[i].getQuality() != sv[i].getQuality())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT qualities do not match: " << sv[i].getQuality()
			 << " != " << cv[i].getQuality() << std::endl;
	     }
	   if(cv[i].getPattern() != sv[i].getPattern())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT patterns do not match: " << sv[i].getPattern()
			 << " != " << cv[i].getPattern() << std::endl;
	     }
	   if(cv[i].getStripType() != sv[i].getStripType())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT strip types do not match: " << sv[i].getStripType()
			 << " != " << cv[i].getStripType() << std::endl;
	     }
	   if(cv[i].getBend() != sv[i].getBend())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT bends do not match: " << sv[i].getBend()
			 << " != " << cv[i].getBend() << std::endl;
	     }
	   if(cv[i].getCFEB() != sv[i].getCFEB())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT CFEBs do not match: " << sv[i].getCFEB()
			 << " != " << cv[i].getCFEB() << std::endl;
	     }
	   if(((short)cv[i].getBX()) != ((short)sv[i].getBX()) - 4)
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT BX do not match: " << sv[i].getBX() - 4
			 << " != " << cv[i].getBX() << std::endl;
	     }
	   if(cv[i].getFullBX() != sv[i].getFullBX())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT Full BX do not match: " << sv[i].getFullBX()
			 << " != " << cv[i].getFullBX() << std::endl;
	     }
	   if(cv[i].getTrknmb() != sv[i].getTrknmb())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "CLCT Track numbers do not match: " << sv[i].getTrknmb()
			 << " != " << cv[i].getTrknmb() << std::endl;
	     }	   
	 }
     }
   for(al = alcts.begin(); al != alcts.end(); ++al)
     {
       if(al->second.first.size() != al->second.second.size())
	 {
	   std::cout << "Major error! # of ALCT digis in detID: " << al->first
		     << " is not equal between sim and unpacked!" << std::endl;
	   //eventually do more in this case!

	   std::vector<CSCALCTDigi> a = al->second.second;
	   std::vector<CSCALCTDigi> b = al->second.first;
	   std::cout << "SIM OUTPUT:" << std::endl;
	   for(std::vector<CSCALCTDigi>::const_iterator i = a.begin(); i != a.end(); ++i)
	     i->print();
	   std::cout << "UNPACKER OUTPUT:" << std::endl;
	   for(std::vector<CSCALCTDigi>::const_iterator i = b.begin(); i != b.end(); ++i)
	     i->print();	     
	 }
       int max = std::min(al->second.first.size(),al->second.second.size());
       std::vector<CSCALCTDigi> cv = al->second.first;
       std::vector<CSCALCTDigi> sv = al->second.second;
       for(int i = 0; i < max; ++i)
	 {
	   if(cv[i].getKeyWG() != sv[i].getKeyWG())
	     {
	       std::cout << "In detId: " << al->first << std::endl;
	       std::cout << "ALCT key wire groups do not match: " << sv[i].getKeyWG()
			 << " != " << cv[i].getKeyWG() << std::endl;
	     }
	   if(cv[i].isValid() != sv[i].isValid())
	     {
	       std::cout << "In detId: " << al->first << std::endl;
	       std::cout << "ALCT Valid bits do not match: " << sv[i].isValid() 
			 << " != " << cv[i].isValid() << std::endl;
	     }
	   if(cv[i].getQuality() != sv[i].getQuality())
	     {
	       std::cout << "In detId: " << al->first << std::endl;
	       std::cout << "ALCT qualities do not match: " << sv[i].getQuality()
			 << " != " << cv[i].getQuality() << std::endl;
	     }
	   if(cv[i].getAccelerator() != sv[i].getAccelerator())
	     {
	       std::cout << "In detId: " << al->first << std::endl;
	       std::cout << "ALCT accelerator bits do not match: " << sv[i].getAccelerator()
			 << " != " << cv[i].getAccelerator() << std::endl;
	     }
	   if(cv[i].getCollisionB() != sv[i].getCollisionB())
	     {
	       std::cout << "In detId: " << al->first << std::endl;
	       std::cout << "ALCT CollisionB flags do not match: " << sv[i].getCollisionB()
			 << " != " << cv[i].getCollisionB() << std::endl;
	     }
	   if((cv[i].getBX()) != (sv[i].getBX()))
	     {
	       std::cout << "In detId: " << al->first << std::endl;
	       std::cout << "ALCT BX do not match: " << sv[i].getBX()
			 << " != " << cv[i].getBX() << std::endl;
	     }
	   if(cv[i].getFullBX() != sv[i].getFullBX())
	     {
	       std::cout << "In detId: " << cl->first << std::endl;
	       std::cout << "ALCT Full BX do not match: " << sv[i].getFullBX()
			 << " != " << cv[i].getFullBX() << std::endl;
	     }
	 }
     }
   for(lc = lcts.begin(); lc != lcts.end(); ++lc)
     {
       if(lc->second.first.size() != lc->second.second.size())
	 {
	   std::cout << "Major error! # of Correlated LCT digis in detID: " << lc->first
		     << " is not equal between sim and unpacked!" << std::endl;
	   //eventually do more in this case!

	   std::vector<CSCCorrelatedLCTDigi> a = lc->second.second;
	   std::vector<CSCCorrelatedLCTDigi> b = lc->second.first;
	   std::cout << "SIM OUTPUT:" << std::endl;
	   for(std::vector<CSCCorrelatedLCTDigi>::const_iterator i = a.begin(); i != a.end(); ++i)
	     i->print();
	   std::cout << "UNPACKER OUTPUT:" << std::endl;
	   for(std::vector<CSCCorrelatedLCTDigi>::const_iterator i = b.begin(); i != b.end(); ++i)
	     i->print();	     
	 }
       int max = std::min(lc->second.first.size(),lc->second.second.size());
       std::vector<CSCCorrelatedLCTDigi> cv = lc->second.first;
       std::vector<CSCCorrelatedLCTDigi> sv = lc->second.second;
       for(int i = 0; i < max; ++i)
	 {
	   if(cv[i].getStrip() != sv[i].getStrip())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT strips do not match: " << sv[i].getStrip()
			 << " != " << cv[i].getStrip() << std::endl;
	     }	 
	   if(cv[i].getKeyWG() != sv[i].getKeyWG())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT key wire groups do not match: " << sv[i].getKeyWG()
			 << " != " << cv[i].getKeyWG() << std::endl;
	     }
	   if(cv[i].isValid() != sv[i].isValid())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT Valid bits do not match: " << sv[i].isValid() 
			 << " != " << cv[i].isValid() << std::endl;
	     }
	   if(cv[i].getQuality() != sv[i].getQuality())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT qualities do not match: " << sv[i].getQuality()
			 << " != " << cv[i].getQuality() << std::endl;
	     }
	   if(cv[i].getPattern() != sv[i].getPattern())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT ALCT patterns do not match: " << sv[i].getPattern()
			 << " != " << cv[i].getPattern() << std::endl;
	     }
	   if(cv[i].getCLCTPattern() != sv[i].getCLCTPattern())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT CLCT patterns do not match: " << sv[i].getCLCTPattern()
			 << " != " << cv[i].getCLCTPattern() << std::endl;
	     }
	   if(cv[i].getStripType() != sv[i].getStripType())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT strip types do not match: " << sv[i].getStripType()
			 << " != " << cv[i].getStripType() << std::endl;
	     }
	   if(cv[i].getBend() != sv[i].getBend())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT bends do not match: " << sv[i].getBend()
			 << " != " << cv[i].getBend() << std::endl;
	     }
	   if(cv[i].getMPCLink() != sv[i].getMPCLink())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT MPC Links do not match: " << sv[i].getMPCLink()
			 << " != " << cv[i].getMPCLink() << std::endl;
	     }
	   if((cv[i].getBX()) != (sv[i].getBX()-6))
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT BX do not match: " << sv[i].getBX()-6
			 << " != " << cv[i].getBX() << std::endl;
	     }
	   if(cv[i].getCSCID() != sv[i].getCSCID())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT CSCIDs do not match: " << sv[i].getCSCID()
			 << " != " << cv[i].getCSCID() << std::endl;
	     }
	   if(cv[i].getBX0() != sv[i].getBX0())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT BX0s do not match: " << sv[i].getBX0()
			 << " != " << cv[i].getBX0() << std::endl;
	     }
	   if(cv[i].getSyncErr() != sv[i].getSyncErr())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT SyncErrs do not match: " << sv[i].getSyncErr()
			 << " != " << cv[i].getSyncErr() << std::endl;
	     }
	   if(cv[i].getTrknmb() != sv[i].getTrknmb())
	     {
	       std::cout << "In detId: " << lc->first << std::endl;
	       std::cout << "Correlated LCT Track numbers do not match: " << sv[i].getTrknmb()
			 << " != " << cv[i].getTrknmb() << std::endl;
	     }	   
	 }
     }
   if(tracks.get().size() != simtracks.get().size())
     {
       std::cout << "Major error! # of L1 Tracks is not equal between sim and unpacked!" << std::endl;
       std::vector<csc::L1Track> a = simtracks.get();
       std::vector<csc::L1Track> b = tracks.get();
       std::cout << "SIM OUTPUT:" << std::endl;
       for(std::vector<csc::L1Track>::const_iterator i = a.begin(); i != a.end(); ++i)
	 i->print();
       std::cout << "UNPACKER OUTPUT:" << std::endl;
       for(std::vector<csc::L1Track>::const_iterator i = b.begin(); i != b.end(); ++i)
	 i->print();
     }
   

   //   iEvent.put(errors);
   return _err;
}

// this function takes the sim wire digis and combines wire digis from the same wire
// into one wire digi, as in the data.
// returns a vector of the combined wire digis
std::vector<CSCWireDigi> 
CSCDigiValidator::sanitizeWireDigis(std::vector<CSCWireDigi>::const_iterator b,
				    std::vector<CSCWireDigi>::const_iterator e)
{
  typedef std::map<int,std::vector<CSCWireDigi> > wire2digi;

  std::vector<CSCWireDigi> _r; // the resulting vector of wire digis
  wire2digi _wr2digis; // map of wires to a set of digis

  for(std::vector<CSCWireDigi>::const_iterator i = b; i != e; ++i)
    _wr2digis[i->getWireGroup()].push_back(*i);

  for(wire2digi::const_iterator i = _wr2digis.begin(); i != _wr2digis.end(); ++i)
    {
      int wire = i->first;
      unsigned tbin = 0x0;

      for(std::vector<CSCWireDigi>::const_iterator d = i->second.begin();
	  d != i->second.end(); ++d)
	{
	  std::vector<int> binson = d->getTimeBinsOn();
	  for(std::vector<int>::const_iterator t = binson.begin(); 
	      t != binson.end(); ++t)
	    tbin |= 1<<(*t);
	}
      
      _r.push_back(CSCWireDigi(wire,tbin));
    }

  return _r;
}

std::vector<CSCStripDigi> 
CSCDigiValidator::relabelStripDigis(const CSCChamberMap* m, CSCDetId _id,
				    std::vector<CSCStripDigi>::const_iterator b,
				    std::vector<CSCStripDigi>::const_iterator e)
{
  std::vector<CSCStripDigi> _r; // the vector of strip digis with appropriate strip #'s

  //bool me1a = _id.station()==1 && _id.ring()==4;
  //bool zplus = _id.endcap()==1;
  //bool me1b = _id.station()==1 && _id.ring()==1;

  for(std::vector<CSCStripDigi>::const_iterator i = b; i != e; ++i)
  {
    int strip=i->getStrip();
    
    //if(me1a&&zplus) strip=17-strip;
    //if(me1b&&!zplus) strip=(65-strip-1)%(m->dmb(_id)*16) + 1;
    //if(me1a) strip+=64;
    
    _r.push_back(CSCStripDigi(strip,i->getADCCounts(),i->getADCOverflow(),
			     i->getOverlappedSample(),i->getErrorstat()));
  }
  return _r;
}

std::vector<CSCComparatorDigi> 
CSCDigiValidator::relabelCompDigis(const CSCChamberMap* m, CSCDetId _id,
				    std::vector<CSCComparatorDigi>::const_iterator b,
				    std::vector<CSCComparatorDigi>::const_iterator e)
{
  std::vector<CSCComparatorDigi> _r; // the vector of comp digis with appropriate strip #'s

  bool me1a = _id.station()==1 && _id.ring()==4;
  //bool zplus = _id.endcap()==1;
  //bool me1b = _id.station()==1 && _id.ring()==1;

  for(std::vector<CSCComparatorDigi>::const_iterator i = b; i != e; ++i)
  {
    int strip=i->getStrip();
    
    //    if(me1a&&zplus) strip=17-strip;
    //    if(me1b&&!zplus) strip=65-strip;
    if(me1a) strip+=64;
    
    _r.push_back(CSCComparatorDigi(strip,i->getComparator(),
				   i->getTimeBinWord()));
  }
  return _r;
}

std::vector<CSCStripDigi> 
CSCDigiValidator::sanitizeStripDigis(std::vector<CSCStripDigi>::const_iterator b,
				     std::vector<CSCStripDigi>::const_iterator e)
{
  std::vector<CSCStripDigi> _r; // vector of digis in proper order

  return _r;
}

std::vector<CSCStripDigi>
CSCDigiValidator::zeroSupStripDigis(std::vector<CSCStripDigi>::const_iterator b,
				    std::vector<CSCStripDigi>::const_iterator e)
{
  std::vector<CSCStripDigi> _r; // zero-suppressed strip digis
  std::vector<int> counts;

  for(std::vector<CSCStripDigi>::const_iterator i = b; i != e; ++i)
    {
      bool nonzero=false;
      counts = i->getADCCounts();
      for(std::vector<int>::const_iterator a = counts.begin(); a != counts.end(); ++a)
	if((*a) != 0) nonzero = true;

      if(nonzero) _r.push_back(*i);
    }

  return _r;
}

// remove comparator digis on or after the 10th time bin, for now, will be configurable later.
std::vector<CSCComparatorDigi>
CSCDigiValidator::zeroSupCompDigis(std::vector<CSCComparatorDigi>::const_iterator b,
				   std::vector<CSCComparatorDigi>::const_iterator e)
{
  std::vector<CSCComparatorDigi> _r;

  for(std::vector<CSCComparatorDigi>::const_iterator i = b; i != e; ++i)
    {
      bool present = false;

      if(i->getTimeBin() < 10) present=true;

      if(present) _r.push_back(*i);
    }

  return _r;
}

// ------------ method called once each job just before starting event loop  ------------
void 
CSCDigiValidator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
CSCDigiValidator::endJob() 
{
}
