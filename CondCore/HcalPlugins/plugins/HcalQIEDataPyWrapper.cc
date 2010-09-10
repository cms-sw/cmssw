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
	public:
		HcalQIEDataDataRepr(unsigned int total, HcalQIEData::tAllContWithNames const & allCont)
			:ADataRepr(total), allContainers(allCont){}



	protected:
		HcalQIEData::tAllContWithNames allContainers;

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
					value = (*contIter).getValue(id);
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
		unsigned int totalValues = 32;
		// get all containers with names
		HcalQIEData::tAllContWithNames allContainers = object().getAllContainers();

		// initializing iterators
		HcalQIEData::tAllContWithNames::const_iterator iter;
		std::vector<HcalQIECoder>::const_iterator contIter;

		ss << "Total HCAL containers: " << allContainers.size() << std::endl;

		typedef std::pair<std::pair< std::vector<float>, std::vector<float> >, int> tPora;

		std::vector<tPora> vec(allContainers.size());

		std::vector<tPora>::iterator iMaz = vec.begin();

		float sum = 0.0, average = 0.0, std_dev = 0.0, sqr_sum = 0.0;
		int size = 0;

		//Run trough all 8 detector containers:
		for (iter = allContainers.begin(), iMaz = vec.begin(); iter != allContainers.end(); ++iter, ++iMaz){
			ss << "---------------------------------------------" << std::endl;
			ss << "Detector: " << (*iter).first << ";    Total values: "<< (*iter).second.size() << std::endl;
			unsigned int j = 0;
			iMaz->second = (*iter).second.size();
			(iMaz->first).first = std::vector<float>(totalValues, 0.0);
			(iMaz->first).second = std::vector<float>(totalValues, 0.0);
			float capValue = 0.0;
			//Run trough all values in container
			for (contIter = (*iter).second.begin(); contIter != (*iter).second.end(); ++contIter){

				//Run trough all values in object:
				for (unsigned int i = 0; i < totalValues; ++i){
					capValue = (*contIter).getValue(i);
					(iMaz->first).first[i] += capValue;
					(iMaz->first).second[i]+= (capValue * capValue);
					//ss << "[" << i << "] " << capValue << ", " << (iMaz->first).first[i] << ", " << (iMaz->first).second[i] << "; ";
				}
				//ss << std::endl;
				++j;
			}



			size = (*iMaz).second;
			unsigned int formatted_nr = 0;
			for (unsigned int i = 0; i < totalValues; ++i){
				sum = ((*iMaz).first).first[i];
				sqr_sum = ((*iMaz).first).second[i];
				average = sum/size;
				//here needs to take absolute value for sqrt:
				std_dev = sqrt( fabs((sqr_sum / size) - (average * average)) );

				//QIEDataCounter(i, formatted_nr);

				if (i >= 16){
				ss  << "    Offset" << QIEDataCounter(i, formatted_nr) << " :"<< std::endl;
				ss	<< "          Average: " << average << "; "<< std::endl;
				ss	<< "          Standart deviation: " << std_dev << "; " << std::endl;					
				}
				else {
				ss  << "    Slope" << QIEDataCounter(i, formatted_nr) << " :"<< std::endl;
				ss	<< "          Average: " << average << "; "<< std::endl;
				ss	<< "          Standart deviation: " << std_dev << "; " << std::endl;
				}

			}	
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
		std::string name = "_Offset_";
		datarepr.nr = 0;
		datarepr.id = 0;
		datarepr.rootname.str("_Offsetrootvalue_");
		datarepr.plotname.str("Offset ");
		datarepr.filename.str("");
		datarepr.filename << filename << name;

		typedef std::vector<TH2F> graphData;
		std::vector< graphData > graphDataVec(numOfValues);
		std::vector< graphData >::iterator imageIter;

		//create images:
		for (imageIter = graphDataVec.begin(); imageIter != graphDataVec.end(); ++imageIter){

			QIEDataCounter(datarepr.id, datarepr.nr);
			
			if (datarepr.id == 16){
				name = "_Slope_";
				datarepr.rootname.str("_Sloperootname_");
				datarepr.plotname.str("Slope ");
				datarepr.filename.str("");
				datarepr.filename << filename << name;					
			}

			if (datarepr.nr == 0){
				datarepr.filename.str("");
				datarepr.filename << filename << name  <<"0";
			} else if ( datarepr.nr == 10)
			{
				datarepr.filename.str("");
				datarepr.filename << filename << name;
			}
			//MAIN FUNCTION:
			datarepr.fillOneGain((*imageIter));

			++(datarepr.nr);
			++(datarepr.id);
		}
		return filename;
	}
}
PYTHON_WRAPPER(HcalQIEData,HcalQIEData);