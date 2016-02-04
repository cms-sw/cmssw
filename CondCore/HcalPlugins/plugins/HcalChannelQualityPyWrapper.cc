#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
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


//functions for correct representation of data in summary and plot:
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"
using namespace HcalObjRepresent;

namespace cond {

	template<>
	class ValueExtractor<HcalChannelQuality>: public  BaseValueExtractor<HcalChannelQuality> {
	public:
		typedef HcalChannelQuality Class;
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
	std::string PayLoadInspector<HcalChannelQuality>::summary() const {
		std::stringstream ss;

		//setting map for representing errors
		std::string statusBitArray[20]; 
		short unsigned int bitMap[9];
		statusBitArray[0] = std::string("cell is off" );
		statusBitArray[1] = std::string("cell is masked/to be masked at RecHit Level" );
		statusBitArray[5] = std::string("cell is dead (from DQM algo)");
		statusBitArray[6] = std::string("cell is hot (from DQM algo)" );
		statusBitArray[7] = std::string("cell has stability error");
		statusBitArray[8] = std::string("cell has timing error" );
		statusBitArray[15] = std::string("cell is masked from the Trigger ");
		statusBitArray[18] = std::string("cell is always excluded from the CaloTower regardless of other bit settings.");
		statusBitArray[19] = std::string("cell is counted as problematic within the tower.");
		bitMap[0] = 0; //{0, 1, 5, 6, 7, 8, 15, 18, 19};
		bitMap[1] = 1;
		bitMap[2] = 5;
		bitMap[3] = 6;
		bitMap[4] = 7;
		bitMap[5] = 8;
		bitMap[6] = 15;
		bitMap[7] = 18;
		bitMap[8] = 19;

		// get all containers with names
		HcalChannelQuality::tAllContWithNames allContainers = object().getAllContainers();

		// initializing iterators
		HcalChannelQuality::tAllContWithNames::const_iterator iter;
		std::vector<HcalChannelStatus>::const_iterator contIter;
		ss << "Total HCAL containers: " << allContainers.size() << std::endl;

		//run trough all pair containers, print error values if any.
		for (iter = allContainers.begin(); iter != allContainers.end(); ++iter){
			ss << "---------------------------------------------" << std::endl;
			ss << "Detector: " << (*iter).first << ";    Total values: "<< (*iter).second.size() << std::endl;
			unsigned int j = 0;
			for (contIter = (*iter).second.begin(); contIter != (*iter).second.end(); ++contIter){

				//if not 0, it have error, print it:
				if ((*contIter).getValue() != 0){//HcalDetId::HcalDetId(uint32_t rawid)
					ss << "     Id["<< j << "]: " <<  
						" rawId: " << (uint32_t)(*contIter).rawId() << " "<< HcalDetId((uint32_t)(*contIter).rawId())<<"; Channel bits: " <<
						(uint32_t)(*contIter).getValue()<< "; Binary format: " << IntToBinary((uint32_t)(*contIter).getValue()) << "; Errors: "
						<< getBitsSummary((uint32_t)((*contIter).getValue()), statusBitArray, bitMap);
				}
				++j;
			}
		}
		return ss.str();
	}

	template<>
	std::string PayLoadInspector<HcalChannelQuality>::plot(std::string const & filename,//
		std::string const &,
		std::vector<int> const&,
		std::vector<float> const& ) const 
	{
		std::vector<TH2F> graphData;
		setup(graphData, "ChannelStatus"); 

		std::stringstream x;
		// Change the titles of each individual histogram
		for (unsigned int d=0;d < graphData.size();++d){
			graphData[d].Reset();
			x << "1+log2(status) for HCAL depth " << d+1;

			//BUG CAN BE HERE:
			//if (ChannelStatus->depth[d]) 
			graphData[d].SetTitle(x.str().c_str());  // replace "setTitle" with "SetTitle", since you are using TH2F objects instead of MonitorElements
			x.str("");
		}

		HcalDetId hcal_id;
		int ieta, depth, iphi, channelBits;
		double logstatus;

		//main loop
		// get all containers with names
		HcalChannelQuality::tAllContWithNames allContainers = object().getAllContainers();

		// initializing iterators
		HcalChannelQuality::tAllContWithNames::const_iterator iter;
		std::vector<HcalChannelStatus>::const_iterator contIter;

		//run trough all pair containers
		for (iter = allContainers.begin(); iter != allContainers.end(); ++iter){
			for (contIter = (*iter).second.begin(); contIter != (*iter).second.end(); ++contIter){
				hcal_id = HcalDetId((uint32_t)(*contIter).rawId());

				channelBits = (uint32_t)(*contIter).getValue();
				if (channelBits == 0) 
					continue;

				depth = hcal_id.depth();
				if (depth<1 || depth>4) 
					continue;

				ieta=hcal_id.ieta();
				iphi=hcal_id.iphi();

				if (hcal_id.subdet() == HcalForward)
					ieta>0 ? ++ieta : --ieta;

				logstatus = log2(1.*channelBits)+1;
				//FILLING GOES HERE:
				graphData[depth-1].Fill(ieta,iphi,logstatus);

				//FOR DEBUGGING:
				//std::cout << "ieta: " << ieta << "; iphi: " << iphi << "; logstatus: " << logstatus << "; channelBits: " << channelBits<< std::endl;
			}
		}
		FillUnphysicalHEHFBins(graphData);



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


		std::stringstream ss;
		ss <<filename << ".png";

		canvas.SaveAs((ss.str()).c_str());

		return (ss.str()).c_str();
	}


}
PYTHON_WRAPPER(HcalChannelQuality,HcalChannelQuality);
