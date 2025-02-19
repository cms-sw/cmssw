#include "CondFormats/HcalObjects/interface/HcalGains.h"

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
	class ValueExtractor<HcalGains>: public  BaseValueExtractor<HcalGains> {
	public:
		typedef HcalGains Class;
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


	template<>
	std::string PayLoadInspector<HcalGains>::summary() const {
		std::stringstream ss;

		// get all containers with names
		HcalGains::tAllContWithNames allContainers = object().getAllContainers();

		// initializing iterators
		HcalGains::tAllContWithNames::const_iterator iter;
		std::vector<HcalGain>::const_iterator contIter;

		ss << "Total HCAL containers: " << allContainers.size() << std::endl;

		int capIdSize = 4;
		float capValue = 0.0;
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
			//ss << sizes.size() <<index;
			//vec.push_back(std::pair<std::vector<float>(4, 0.0), 0>);
			iMaz->second = (*iter).second.size();
			(iMaz->first).first = std::vector<float>(4, 0.0);
			(iMaz->first).second = std::vector<float>(4, 0.0);

			//Run trough all values in container
			for (contIter = (*iter).second.begin(); contIter != (*iter).second.end(); ++contIter){

				//Run trough all values (4) in object:
				for (int i = 0; i < capIdSize; ++i){
					capValue = (*contIter).getValue(i);
					(iMaz->first).first[i] += capValue;
					(iMaz->first).second[i]+= (capValue * capValue);
					//ss << "[" << i << "] " << capValue << ", " << (iMaz->first).first[i] << ", " << (iMaz->first).second[i] << "; ";
				}
				//ss << std::endl;
				++j;
			}

			size = (*iMaz).second;
			//ss << k++ << ": size =  "<< size << "; ";
			for (int i = 0; i < capIdSize; ++i){
				sum = ((*iMaz).first).first[i];
				sqr_sum = ((*iMaz).first).second[i];
				average = sum/size;
				std_dev = sqrt( (sqr_sum/size) - (average * average) );
				ss  << "    Gain " << i << " :"<< std::endl;
				ss	<< "          Average: " << average << "; "<< std::endl;
				ss	<< "          Standart deviation: " << std_dev << "; " << std::endl;
			}	
		}
		return ss.str();
	}

	void draw(std::vector<TH2F> &graphData, std::string filename) {
		//Drawing...
		// use David's palette
		gStyle->SetPalette(1);
		const Int_t NCont = 999;
		gStyle->SetNumberContours(NCont);
		TCanvas canvas("CC map","CC map",840,369*4);

		TPad pad1("pad1","pad1", 0.0, 0.75, 1.0, 1.0);
		pad1.Draw();
		TPad pad2("pad2","pad2", 0.0, 0.5, 1.0, 0.75);
		pad2.Draw();
		TPad pad3("pad3","pad3", 0.0, 0.25, 1.0, 0.5);
		pad3.Draw();
		TPad pad4("pad4","pad4", 0.0, 0.0, 1.0, 0.25);
		pad4.Draw();


		pad1.cd();
		graphData[0].SetStats(0);
		graphData[0].Draw("colz");

		pad2.cd();
		graphData[1].SetStats(0);
		graphData[1].Draw("colz");

		pad3.cd();
		graphData[2].SetStats(0);
		graphData[2].Draw("colz");

		pad4.cd();
		graphData[3].SetStats(0);
		graphData[3].Draw("colz");

		canvas.SaveAs(filename.c_str());
	}

	template<>
	std::string PayLoadInspector<HcalGains>::plot(std::string const & filename,
		std::string const &,
		std::vector<int> const&,
		std::vector<float> const& ) const 
	{
		 //void set(since, till, filename); {
		unsigned int numOfValues = 4;
		std::string plotname = "_Gain_";
		//filename = filename + "_Gain";


		typedef std::vector<TH2F> graphData;
		std::vector< graphData > graphDataVec(numOfValues);
		std::vector< graphData >::iterator bigIter;

		std::stringstream ss, plotname_i;
		HcalGains::tAllContWithNames allContainers = object().getAllContainers();

		int i = 0;
		for (bigIter = graphDataVec.begin(); bigIter != graphDataVec.end(); ++bigIter){
			plotname_i << plotname << i ;
			ss << filename << "_Gain_"<<i<<".png";
			fillOneGain((*bigIter), allContainers, plotname_i.str(), i);
			FillUnphysicalHEHFBins(*bigIter);
			draw((*bigIter), ss.str());
			++i;
			ss.str("");
			plotname_i.str("");
		}
		return filename;
	}
}
PYTHON_WRAPPER(HcalGains,HcalGains);