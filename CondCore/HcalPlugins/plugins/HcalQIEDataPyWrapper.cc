#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>
#include <sstream>

#include "TH1F.h"
#include "TH2F.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"

#include "math.h"
//functions for correct representation of data in summary and plot:
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"
using namespace HcalObjRepresent;

namespace cond {
	template<>
	class ValueExtractor<HcalQIEData>: public  BaseValueExtractor<HcalQIEData> {
	public:
		typedef HcalQIEData Class;
		typedef ExtractWhat<Class> What;
		static What what() { return What();}

		ValueExtractor(){}

		ValueExtractor(What const & what)
		{
			// here one can make stuff really complicated...
		}

		void compute(Class const & it){
		}
	private:

	};

	class HcalQIEDataDataRepr: public ADataRepr
	{
	protected:
		HcalQIEData::tAllContWithNames allContainers;
	public:
		unsigned int fCapId;
		unsigned int fRange;
		bool slopeOrOffset;

		HcalQIEDataDataRepr(unsigned int total, HcalQIEData::tAllContWithNames const & allCont)
			:ADataRepr(total), allContainers(allCont), fCapId(0), fRange(0), slopeOrOffset(false){}




	protected:
		void doFillIn(std::vector<TH2F> &graphData){
			//ITERATORS AND VALUES:
			HcalQIEData::tAllContWithNames::const_iterator iter;
			std::vector<HcalQIECoder>::const_iterator contIter;
			float value = 0.0;

			//run trough all pair containers
			for (iter = allContainers.begin(); iter != allContainers.end(); ++iter){
				//Run trough all values:
				for (contIter = (*iter).second.begin(); contIter != (*iter).second.end(); ++contIter){
					hcal_id = HcalDetId((uint32_t)(*contIter).rawId());

					depth = hcal_id.depth();
					if (depth<1 || depth>4) 
						continue;

					ieta=hcal_id.ieta();
					iphi=hcal_id.iphi();

					if (hcal_id.subdet() == HcalForward)
						ieta>0 ? ++ieta : --ieta;

					//GET VALUE:
					//unsigned fCapId, unsigned fRange;
					if (slopeOrOffset){
						value = (*contIter).slope(fCapId, fRange);
					} else {
						value = (*contIter).offset(fCapId, fRange);
					}
					//logstatus = log2(1.*channelBits)+1;

					//FILLING GOES HERE:
					graphData[depth-1].Fill(ieta,iphi, value);	
				}
			}
		}
	};

	std::string QIEDataCounter(const int nr, unsigned int &formated_nr, int base = 4){
		int numer = nr;
		int tens = 0, ones = 0;
		if (numer >= 16){
			numer = numer -16;
		}
		tens = numer / base;
		ones = numer - (tens*base);
		formated_nr = tens*10 + ones;
		std::stringstream ss;
		ss << tens << ones;
		return ss.str();
	}

	template<>
	std::string PayLoadInspector<HcalQIEData>::summary() const {
		std::stringstream ss;
		// get all containers with names
		HcalQIEData::tAllContWithNames allContainers = object().getAllContainers();

		// initializing iterators
		HcalQIEData::tAllContWithNames::const_iterator iter;
		std::vector<HcalQIECoder>::const_iterator contIter;

		ss << "Total HCAL containers: " << allContainers.size() << std::endl;

		typedef std::pair<std::pair< float, float >, int> tPora;

		//std::vector<tPora> vMaz(allContainers.size());
		//std::vector<tPora>::iterator iMaz = vMaz.begin();

		std::vector<tPora> vOffsets(allContainers.size());// for each container (total 8)
		std::vector<tPora> vSlopes(allContainers.size());// for each container (total 8)

		std::vector<tPora>::iterator iOffset = vOffsets.begin();
		std::vector<tPora>::iterator iSlope = vSlopes.begin();

		float offset = .0;
		float slope = .0;


		//Run trough all 8 detector containers:
		for (iter = allContainers.begin(); iter != allContainers.end(); ++iter, ++iOffset, ++iSlope){
			iOffset->second = (*iter).second.size();// total number of values in detector
			iSlope->second = (*iter).second.size();// total number of values in detector

			(iOffset->first).first = .0;
			(iOffset->first).second = .0;
			(iSlope->first).first = .0;
			(iSlope->first).second = .0;

			//Run trough all values in container
			unsigned int i = 0;
			for (contIter = (*iter).second.begin(); contIter != (*iter).second.end(); ++contIter){

				//Run trough all values in object:
				for (unsigned int fCapId = 0; fCapId < 4; ++fCapId){
					for (unsigned int fRange = 0; fRange < 4; ++fRange){
						offset =  (*contIter).offset (fCapId, fRange);
						(iOffset->first).first += offset;
						(iOffset->first).second+= (offset * offset);

						slope = (*contIter).slope (fCapId, fRange);
						(iSlope->first).first += slope;
						(iSlope->first).second+= (slope * slope);
					}
				}
				++i;
				//ss << "[" << i << "] " << capValue << ", " << (iMaz->first).first[i] << ", " << (iMaz->first).second[i] << "; ";
			}
		}
		//ss << std::endl;

		//got all the values, now do the work:
		iOffset = vOffsets.begin();
		iSlope = vSlopes.begin();
		float sumOffset = 0.0, averageOffset = 0.0, std_devOffset = 0.0, sqr_sumOffset = 0.0;
		int sizeOffset = 0;

		float sumSlope = 0.0, averageSlope = 0.0, std_devSlope = 0.0, sqr_sumSlope = 0.0;
		int sizeSlope = 0;

		sizeOffset = (*iOffset).second;
		sizeSlope = (*iSlope).second;


		
		unsigned int i = 0;
		for (iter = allContainers.begin(); iter != allContainers.end(); ++iter, ++i, ++iSlope, ++iOffset){

			ss << "---------------------------------------------" << std::endl;
			ss << "Detector: " << (*iter).first << ";    Total values: "<< (*iter).second.size() << std::endl;
			sumOffset = ((*iOffset).first).first;
			sqr_sumOffset = ((*iOffset).first).second;
			averageOffset = sumOffset/sizeOffset;
			//here needs to take absolute value for sqrt:
			std_devOffset = sqrt( fabs((sqr_sumOffset / sizeOffset) - (averageOffset * averageOffset)) );

			sumSlope = ((*iSlope).first).first;
			sqr_sumSlope = ((*iSlope).first).second;
			averageSlope = sumSlope/sizeSlope;
			//here needs to take absolute value for sqrt:
			std_devSlope = sqrt( fabs((sqr_sumSlope / sizeSlope) - (averageSlope * averageSlope)) );

			ss  << "    Offset: " << std::endl;
			ss	<< "          Average: " << averageOffset << "; "<< std::endl;
			ss	<< "          Standart deviation: " << std_devOffset << "; " << std::endl;					
			ss  << "    Slope: " << std::endl;
			ss	<< "          Average: " << averageSlope << "; "<< std::endl;
			ss	<< "          Standart deviation: " << std_devSlope << "; " << std::endl;
		}		
		//std::cout << ss.str();
		return ss.str();
	}

	template<>
	std::string PayLoadInspector<HcalQIEData>::plot(std::string const & filename,
		std::string const &,
		std::vector<int> const&,
		std::vector<float> const& ) const 
	{

		//how much values are in container
		unsigned int numOfValues = 32;

		//create object helper for making plots;
		HcalQIEDataDataRepr datarepr(numOfValues, object().getAllContainers());

		typedef std::vector<TH2F> graphData;
		std::vector< graphData > graphDataVec(numOfValues);
		std::vector< graphData >::iterator imageIter;
		imageIter = graphDataVec.begin();



		std::string name = "_Offset_";
		datarepr.nr = 0;
		datarepr.id = 0;
		datarepr.rootname.str("_Offsetrootvalue_");
		datarepr.plotname.str("Offset ");
		datarepr.filename.str("");
		datarepr.filename << filename << name;
		//Run trough all values in object:
		datarepr.slopeOrOffset = false;

		for (unsigned int fCapId = 0; fCapId < 4; ++fCapId){
			for (unsigned int fRange = 0; fRange < 4; ++fRange){
				datarepr.fCapId = fCapId;
				datarepr.fRange = fRange;

				QIEDataCounter(datarepr.id, datarepr.nr);

				if (datarepr.nr == 0){
					datarepr.filename.str("");
					datarepr.filename << filename << name  <<"0";
				} else if ( datarepr.nr == 10)
				{
					datarepr.filename.str("");
					datarepr.filename << filename << name;
				}



				datarepr.fillOneGain((*imageIter));


				++(datarepr.id);
				++(datarepr.nr);
				++(imageIter);
			}
		}
/////////////////////////////
		name = "_Slope_";
		datarepr.rootname.str("_Sloperootname_");
		datarepr.plotname.str("Slope ");
		datarepr.filename.str("");
		datarepr.filename << filename << name;	

		datarepr.slopeOrOffset = true;
		datarepr.nr = 0;
		datarepr.id = 0;

		for (unsigned int fCapId = 0; fCapId < 4; ++fCapId){
			for (unsigned int fRange = 0; fRange < 4; ++fRange){
				datarepr.fCapId = fCapId;
				datarepr.fRange = fRange;

				QIEDataCounter(datarepr.id, datarepr.nr);

				if (datarepr.nr == 0){
					datarepr.filename.str("");
					datarepr.filename << filename << name  <<"0";
				} else if ( datarepr.nr == 10)
				{
					datarepr.filename.str("");
					datarepr.filename << filename << name;
				}

				datarepr.fillOneGain((*imageIter));

				++(datarepr.id);
				++(datarepr.nr);
				++(imageIter);
			}
		}


		return filename;
	}
}

PYTHON_WRAPPER(HcalQIEData,HcalQIEData);