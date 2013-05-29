#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
//#include "CondTools/Ecal/interface/EcalTPGFineGrainEBMapXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <fstream>

namespace cond {
	template<>
	std::string PayLoadInspector<EcalTPGTowerStatus>::summary() const {
		std::stringstream ss;
		int ieta = 0;
		int iphi = 0;

		const EcalTPGTowerStatusMap & badTTMap = object().getMap();
		EcalTPGTowerStatusMapIterator it;

		ss <<"Barrel and endcap masked Trigger Towers"<<std::endl;
		//ss <<"RawId " << "     iphi " << "  ieta " << std::endl;
		//ss <<""<< std::endl;

		//for (it=badTTMap.begin();it!=badTTMap.end();++it) {

		//	// Print in the text file only the masked barrel and endcap TTs
		//	if ((*it).second != 0){
		//		EcalTrigTowerDetId  ttId((*it).first);
		//		ieta = ttId.ieta();
		//		iphi = ttId.iphi();
		//		ss <<""<< std::dec<<(*it).first << "  " << iphi << "     " << ieta << std::endl;
		//	}
		//}
			const std::map<uint32_t, uint16_t> intsMap = object().getMap();
			std::map<uint32_t, uint16_t>::const_iterator intsIter;
			std::vector<EcalTrigTowerDetId> towerVec;
			std::vector<EcalTrigTowerDetId>::const_iterator towIter;
			//map keys are rawIds, get EcalTrigTowerDetIds from them:
			ss << "intsMap size: " << intsMap.size() << std::endl;
			for(intsIter = intsMap.begin(); intsIter != intsMap.end(); ++intsIter){
				ss << "Key: " << intsIter->first << "->Value: " << intsIter->second;
				towerVec.push_back(EcalTrigTowerDetId(intsIter->first));
				towIter = towerVec.end() - 1;
				ss
				//<< " ieta: " << towIter->ieta() 
				//<< " iphi: " << towIter->iphi() 
				<< " subdet: " <<  towIter->subDet()
				//<< " zside: " << towIter->zside()
				<< " ietaAbs: " << towIter->ietaAbs()
				//<< " iquadratant: " << towIter->iquadrant()
				<< " hashedIndex: " << towIter->hashedIndex()
				//<< " denseIndex: " << towIter->denseIndex()
				//<< " iDCC: " << towIter->iDCC()
				//<< " iTT: " << towIter->iTT()
				<< " Stream: " << *towIter
				<< std::endl;			
			}
		return ss.str();
	}

	template<>
	std::string PayLoadInspector<EcalTPGTowerStatus>::plot(std::string const & filename,
		std::string const &, 
		std::vector<int> const&, 
		std::vector<float> const& ) const {
			gStyle->SetPalette(1);
			const int TOTAL_IMAGES = 1;
			const int TOTAL_PADS = 3;

			const float IMG_SIZE = 1.5;
			TCanvas canvas("CC map","CC map",800*IMG_SIZE, 200 * TOTAL_IMAGES*IMG_SIZE);//800, 1200

			float xmi[3] = {0.0 , 0.22, 0.78};
			float xma[3] = {0.22, 0.78, 1.00};


			TPad*** pad = new TPad**[TOTAL_IMAGES];
			for (int gId = 0; gId < TOTAL_IMAGES; gId++) {
				pad[gId] = new TPad*[TOTAL_PADS];
				for (int obj = 0; obj < TOTAL_PADS; obj++) {
					float yma = 1. - 0;//1.- (0.17 * gId);
					float ymi = yma - 1;//0.15;
					pad[gId][obj] = new TPad(Form("p_%i_%i", obj, gId),Form("p_%i_%i", obj, gId),
						xmi[obj], ymi, xma[obj], yma);
					pad[gId][obj]->Draw();
				}
			}
			////////
			const std::map<uint32_t, uint16_t> intsMap = object().getMap();
			std::map<uint32_t, uint16_t>::const_iterator intsIter;
			std::vector<EcalTrigTowerDetId> towerVec;
			std::vector<EcalTrigTowerDetId>::const_iterator towIter;
			std::stringstream ss;
			//map keys are rawIds, get EcalTrigTowerDetIds from them:
			std::cout << "intsMap size: " << intsMap.size() << std::endl;
			for(intsIter = intsMap.begin(); intsIter != intsMap.end(); ++intsIter){
				std::cout << "Key: " << intsIter->first << "->Value: " << intsIter->second;
				towerVec.push_back(EcalTrigTowerDetId(intsIter->first));
				towIter = towerVec.end() - 1;
				std::cout 
				//<< " ieta: " << towIter->ieta() 
				//<< " iphi: " << towIter->iphi() 
				<< " subdet: " <<  towIter->subDet()
				//<< " zside: " << towIter->zside()
				<< " ietaAbs: " << towIter->ietaAbs()
				//<< " iquadratant: " << towIter->iquadrant()
				<< " hashedIndex: " << towIter->hashedIndex()
				//<< " denseIndex: " << towIter->denseIndex()
				//<< " iDCC: " << towIter->iDCC()
				//<< " iTT: " << towIter->iTT()
				<< " Stream: " << *towIter
				<< std::endl;			
			}

			//<BUSINESS LOGIC>
			//If in tower exists crystal with status == 1, plot all values in this tower as 1;
			//Run trough towers values
			//</BUSINESS LOGIC>

			//for (towIter = towerVec.begin(); towIter != towerVec.end(); ++towIter){
			//	std::cout 
			//	<< " ieta: " << towIter->ieta() 
			//	<< " iphi: " << towIter->iphi() 
			//	<< " subdet: " <<  towIter->subDet()
			//	<< " zside: " << towIter->zside()
			//	<< " ietaAbs: " << towIter->ietaAbs()
			//	<< " iquadratant: " << towIter->iquadrant()
			//	<< " hashedIndex: " << towIter->hashedIndex()
			//	<< " denseIndex: " << towIter->denseIndex()
			//	//<< " iDCC: " << towIter->iDCC()
			//	//<< " iTT: " << towIter->iTT()
			//	<< " Stream: " << *towIter
			//	<< std::endl;
			//}
			////////
/*
			const int kGains       = 3;
			const int gainValues[3] = {12, 6, 1};
			const int kSides       = 2;
			const int kBarlRings   = EBDetId::MAX_IETA;
			const int kBarlWedges  = EBDetId::MAX_IPHI;
			const int kEndcWedgesX = EEDetId::IX_MAX;
			const int kEndcWedgesY = EEDetId::IY_MAX;

			TH2F** barrel_m = new TH2F*[3];
			TH2F** endc_p_m = new TH2F*[3];
			TH2F** endc_m_m = new TH2F*[3];
			std::string variableName = "TPGTowerStatus";
			for (int gainId = 0; gainId < kGains; gainId++) {
				barrel_m[gainId] = new TH2F(Form((variableName + " EBm%i").c_str(),gainId), Form((variableName + " EB").c_str(),gainValues[gainId]),360,0,360, 170, -85,85);
				endc_p_m[gainId] = new TH2F(Form((variableName + " EE+m%i").c_str(),gainId), Form((variableName + " EE+").c_str(),gainValues[gainId]),100,1,101,100,1,101);
				endc_m_m[gainId] = new TH2F(Form((variableName + " EE-m%i").c_str(),gainId), Form((variableName + " EE-").c_str(),gainValues[gainId]),100,1,101,100,1,101);
			}

			for (int sign=0; sign < kSides; sign++) {
				int thesign = sign==1 ? 1:-1;

				for (int ieta=0; ieta<kBarlRings; ieta++) {
					for (int iphi=0; iphi<kBarlWedges; iphi++) {
						EBDetId id((ieta+1)*thesign, iphi+1);
						float y = -1 - ieta;
						if(sign == 1) y = ieta;
						barrel_m[0]->Fill(iphi, y, object()[id.rawId()].getStatusCode());
					}  // iphi
				}   // ieta

				for (int ix=0; ix<kEndcWedgesX; ix++) {
					for (int iy=0; iy<kEndcWedgesY; iy++) {
						if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
						EEDetId id(ix+1,iy+1,thesign);
						if (thesign==1) {
							endc_p_m[0]->Fill(ix+1,iy+1,object()[id.rawId()].getStatusCode());
						}
						else{ 
							endc_m_m[0]->Fill(ix+1,iy+1,object()[id.rawId()].getStatusCode());
						}
					}  // iy
				}   // ix
			}    // side

			//canvas.cd(1);
			TLine* l = new TLine(0., 0., 0., 0.);
			l->SetLineWidth(1);
			int ixSectorsEE[202] = {
				62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 56, 56, 46, 46, 44, 44, 43, 43, 42, 42, 
				41, 41, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 46, 46, 56, 56, 58, 58, 59, 59, 
				60, 60, 61, 61, 62, 62,  0,101,101, 98, 98, 96, 96, 93, 93, 88, 88, 86, 86, 81, 
				81, 76, 76, 66, 66, 61, 61, 41, 41, 36, 36, 26, 26, 21, 21, 16, 16, 14, 14,  9,
				9,  6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9, 14, 14, 16, 16, 21, 21, 26, 
				26, 36, 36, 41, 41, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 93, 93, 96, 
				96, 98, 98,101,101,  0, 62, 66, 66, 71, 71, 81, 81, 91, 91, 93,  0, 62, 66, 66, 
				91, 91, 98,  0, 58, 61, 61, 66, 66, 71, 71, 76, 76, 81, 81,  0, 51, 51,  0, 44, 
				41, 41, 36, 36, 31, 31, 26, 26, 21, 21,  0, 40, 36, 36, 11, 11,  4,  0, 40, 36, 
				36, 31, 31, 21, 21, 11, 11,  9,  0, 46, 46, 41, 41, 36, 36,  0, 56, 56, 61, 61, 66, 66};

				int iySectorsEE[202] = {
					51, 56, 56, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 61, 61, 60, 60, 59, 59, 58, 
					58, 56, 56, 46, 46, 44, 44, 43, 43, 42, 42, 41, 41, 40, 40, 41, 41, 42, 42, 43, 
					43, 44, 44, 46, 46, 51,  0, 51, 61, 61, 66, 66, 76, 76, 81, 81, 86, 86, 88, 88, 
					93, 93, 96, 96, 98, 98,101,101, 98, 98, 96, 96, 93, 93, 88, 88, 86, 86, 81, 81, 
					76, 76, 66, 66, 61, 61, 41, 41, 36, 36, 26, 26, 21, 21, 16, 16, 14, 14,  9,  9, 
					6,  6,  4,  4,  1,  1,  4,  4,  6,  6,  9,  9, 14, 14, 16, 16, 21, 21, 26, 26, 
					36, 36, 41, 41, 51,  0, 46, 46, 41, 41, 36, 36, 31, 31, 26, 26,  0, 51, 51, 56, 
					56, 61, 61,  0, 61, 61, 66, 66, 71, 71, 76, 76, 86, 86, 88,  0, 62,101,  0, 61, 
					61, 66, 66, 71, 71, 76, 76, 86, 86, 88,  0, 51, 51, 56, 56, 61, 61,  0, 46, 46, 
					41, 41, 36, 36, 31, 31, 26, 26,  0, 40, 31, 31, 16, 16,  6,  0, 40, 31, 31, 16, 16,  6};

					for (int gId = 0; gId < TOTAL_IMAGES; gId++) {//was 3
						pad[gId][0]->cd();
						endc_m_m[gId]->SetStats(0);
						//endc_m_m[gId]->SetMaximum(225);
						//endc_m_m[gId]->SetMinimum(175);
						endc_m_m[gId]->Draw("colz");
						for ( int i=0; i<201; i=i+1) {
							if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
								(ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
									l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
										ixSectorsEE[i+1], iySectorsEE[i+1]);
									l->SetLineWidth(0.2);
							}
						}

						//canvas.cd(2);
						pad[gId][1]->cd();
						barrel_m[gId]->SetStats(0);
						//barrel_m[gId]->SetMaximum(225);
						//barrel_m[gId]->SetMinimum(175);
						barrel_m[gId]->Draw("colz");
						for(int i = 0; i <17; i++) {
							Double_t x = 20.+ (i *20);
							l = new TLine(x,-85.,x,86.);
							l->Draw();
						}
						l = new TLine(0.,0.,360.,0.);
						l->Draw();

						//canvas.cd(3);
						pad[gId][2]->cd();
						endc_p_m[gId]->SetStats(0);
						//endc_p_m[gId]->SetMaximum(225);
						//endc_p_m[gId]->SetMinimum(175);
						endc_p_m[gId]->Draw("colz");
						for ( int i=0; i<201; i=i+1) {
							if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
								(ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
									l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
										ixSectorsEE[i+1], iySectorsEE[i+1]);
							}
						}
					}
					*/

					canvas.SaveAs(filename.c_str());
					return filename;
	}  // plot
}
PYTHON_WRAPPER(EcalTPGTowerStatus,EcalTPGTowerStatus);
