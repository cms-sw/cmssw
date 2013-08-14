#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorr.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>
#include <sstream>

#include "TH1F.h"
#include "TH2F.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "math.h"
//functions for correct representation of data in summary and plot:
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"
using namespace HcalObjRepresent;

namespace cond {
	template<>
	class ValueExtractor<HcalTimeCorrs>: public  BaseValueExtractor<HcalTimeCorrs> {
	public:
		typedef HcalTimeCorrs Class;
		typedef ExtractWhat<Class> What;
		static What what() { return What();}

		ValueExtractor(){}

		ValueExtractor(What const & what)
		{
			// here one can make stuff really complicated...
		}

		void compute(Class const & it) override{
		}
	private:

	};

	class HcalTimeCorrsDataRepr: public ADataRepr
	{
	public:
		HcalTimeCorrsDataRepr(unsigned int total, HcalTimeCorrs::tAllContWithNames const & allCont)
			:ADataRepr(total), allContainers(allCont){}



	protected:
		HcalTimeCorrs::tAllContWithNames allContainers;

		void doFillIn(std::vector<TH2F> &graphData) override{
			//ITERATORS AND VALUES:
			HcalTimeCorrs::tAllContWithNames::const_iterator iter;
			std::vector<HcalTimeCorr>::const_iterator contIter;
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
					value = (*contIter).getValue();
					//logstatus = log2(1.*channelBits)+1;

					//FILLING GOES HERE:
					graphData[depth-1].Fill(ieta,iphi, value);	
				}
			}
		}
	};

	template<>
	std::string PayLoadInspector<HcalTimeCorrs>::summary() const {
		std::stringstream ss;
		unsigned int totalValues = 1;
		// get all containers with names
		HcalTimeCorrs::tAllContWithNames allContainers = object().getAllContainers();

		// initializing iterators
		HcalTimeCorrs::tAllContWithNames::const_iterator iter;
		std::vector<HcalTimeCorr>::const_iterator contIter;

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

				//Run trough all values (8) in object:
				for (unsigned int i = 0; i < totalValues; ++i){
					capValue = (*contIter).getValue();
					(iMaz->first).first[i] += capValue;
					(iMaz->first).second[i]+= (capValue * capValue);
					//ss << "[" << i << "] " << capValue << ", " << (iMaz->first).first[i] << ", " << (iMaz->first).second[i] << "; ";
				}
				//ss << std::endl;
				++j;
			}

			size = (*iMaz).second;
			for (unsigned int i = 0; i < totalValues; ++i){
				sum = ((*iMaz).first).first[i];
				sqr_sum = ((*iMaz).first).second[i];
				average = sum/size;
				//here needs to take absolute value for sqrt:
				std_dev = sqrt( fabs((sqr_sum / size) - (average * average)) );

				ss  << "   TimeCorr " << " :"<< std::endl;
				ss	<< "          Average: " << average << "; "<< std::endl;
				ss	<< "          Standart deviation: " << std_dev << "; " << std::endl;					

			}	
		}
		//std::cout << ss.str();
			return ss.str();
	}

	template<>
	std::string PayLoadInspector<HcalTimeCorrs>::plot(std::string const & filename,
		std::string const &,
		std::vector<int> const&,
		std::vector<float> const& ) const 
	{

		//how much values are in container
		unsigned int numOfValues = 1;

		//create object helper for making plots;
		HcalTimeCorrsDataRepr datarepr(numOfValues, object().getAllContainers());

		datarepr.nr = 0;
		datarepr.id = 0;
		datarepr.rootname.str("_TimeCorrrootvalue_");
		datarepr.plotname.str("TimeCorr ");
		datarepr.filename.str("");
		datarepr.filename << filename << "";

		typedef std::vector<TH2F> graphData;
		std::vector< graphData > graphDataVec(numOfValues);
		std::vector< graphData >::iterator imageIter;

		//create images:
		for (imageIter = graphDataVec.begin(); imageIter != graphDataVec.end(); ++imageIter){
			//MAIN FUNCTION:
			datarepr.fillOneGain((*imageIter));

			++(datarepr.nr);
			++(datarepr.id);
		}
		return filename;
	}
}
PYTHON_WRAPPER(HcalTimeCorrs,HcalTimeCorrs);