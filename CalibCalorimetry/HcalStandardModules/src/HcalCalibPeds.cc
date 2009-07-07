// Original Author:  Steven Won
//         Created:  Fri May  2 15:34:43 CEST 2008
// Written to replace the combination of HcalPedestalAnalyzer and HcalPedestalAnalysis 
// This code runs 1000x faster and produces all outputs from a single run
// (ADC, fC in .txt plus an .xml file)
//
#include <memory>
#include "CalibCalorimetry/HcalStandardModules/interface/HcalCalibPeds.h"

HcalCalibPeds::HcalCalibPeds(const edm::ParameterSet& ps)
{
   firstTS = ps.getUntrackedParameter<int>("firstTS", 0);
   lastTS = ps.getUntrackedParameter<int>("lastTS", 9);   
   pedsADCfilename = ps.getUntrackedParameter<std::string>("fileName","calib_peds.txt");
   firsttime = true;
}


HcalCalibPeds::~HcalCalibPeds()
{
   HcalPedestals* rawPedsItem = new HcalPedestals(true);

   //Calculate pedestal constants
   std::cout << "Calculating Pedestal constants...\n";
   std::vector<CalibPedBunch>::iterator bunch_it;
   for(bunch_it=Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
   {
      if(bunch_it->usedflag){

      //pedestal constant is the mean
      bunch_it->cap[0] /= bunch_it->num[0];
      bunch_it->cap[1] /= bunch_it->num[1];
      bunch_it->cap[2] /= bunch_it->num[2];
      bunch_it->cap[3] /= bunch_it->num[3];

      const HcalPedestal item(bunch_it->calibdetid, bunch_it->cap[0], bunch_it->cap[1], bunch_it->cap[2], bunch_it->cap[3]);
      rawPedsItem->addValues(item);
      }
   }

    // dump the resulting list of pedestals into a file
    std::ofstream outStream1(pedsADCfilename.c_str());
    HcalDbASCIIIO::dumpObject (outStream1, (*rawPedsItem) );
}

// ------------ method called to for each event  ------------
void
HcalCalibPeds::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   edm::Handle<HcalCalibDigiCollection> calib_digi;   e.getByType(calib_digi);
   edm::ESHandle<HcalDbService> conditions;
   iSetup.get<HcalDbRecord>().get(conditions);

   if(firsttime)
   {
      edm::ESHandle<HcalElectronicsMap> refEMap;
      iSetup.get<HcalElectronicsMapRcd>().get(refEMap);
      const HcalElectronicsMap* myRefEMap = refEMap.product();
      std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
      for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
      {     
         HcalGenericDetId mygenid(it->rawId());
         if(mygenid.isHcalCalibDetId())
         {
            CalibPedBunch a;
            HcalCalibDetId chanid(mygenid.rawId());
            a.calibdetid = chanid;
            a.usedflag = false;
            for(int i = 0; i != 4; i++)
            {
               a.num[i] = 0;
               a.cap[i] = 0;
            }
            Bunches.push_back(a);
         }
      }
      firsttime = false;
   }

   std::vector<CalibPedBunch>::iterator bunch_it;

   for(HcalCalibDigiCollection::const_iterator j = calib_digi->begin(); j != calib_digi->end(); j++)
   {
      const HcalCalibDataFrame digi = (const HcalCalibDataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->calibdetid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts != lastTS+1; ts++)
      {
         bunch_it->num[digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
      }
   }

   
//this is the last brace
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalCalibPeds);
