
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondTools/Ecal/interface/EcalChannelStatusXMLTranslator.h"
#include "TROOT.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/iterator/transform_iterator.hpp>

namespace cond {


	namespace ecalcond {

		typedef EcalChannelStatus Container;
		typedef Container::Items  Items;
		typedef Container::value_type  value_type;

		enum How { singleChannel, bySuperModule, barrel, endcap, all};

		int bad(Items const & cont) {
			return  std::count_if(cont.begin(),cont.end(),
				boost::bind(std::greater<int>(),
				boost::bind(&value_type::getStatusCode,_1),0)
				);
		}


		void extractBarrel(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
			result.resize(1);
			result[0] =  bad(cont.barrelItems());
		}
		void extractEndcap(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
			result.resize(1);
			result[0] = bad(cont.endcapItems());
		}
		void extractAll(Container const & cont, std::vector<int> const &,  std::vector<float> & result) {
			result.resize(1);
			result[0] = bad(cont.barrelItems())+bad(cont.endcapItems());
		}

		void extractSuperModules(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
			// bho...
		}

		void extractSingleChannel(Container const & cont, std::vector<int> const & which,  std::vector<float> & result) {
			result.reserve(which.size());
			for (unsigned int i=0; i<which.size();i++) {
				result.push_back(cont[which[i]].getStatusCode());
			}
		}

		typedef boost::function<void(Container const & cont, std::vector<int> const & which,  std::vector<float> & result)> CondExtractor;

	} // ecalcond

	template<>
	struct ExtractWhat<ecalcond::Container> {

		ecalcond::How m_how;
		std::vector<int> m_which;

		ecalcond::How const & how() const { return m_how;}
		std::vector<int> const & which() const { return m_which;}

		void set_how(ecalcond::How i) {m_how=i;}
		void set_which(std::vector<int> & i) { m_which.swap(i);}
	};




	template<>
	class ValueExtractor<ecalcond::Container>: public  BaseValueExtractor<ecalcond::Container> {
	public:

		static ecalcond::CondExtractor & extractor(ecalcond::How how) {
			static  ecalcond::CondExtractor fun[5] = { 
				ecalcond::CondExtractor(ecalcond::extractSingleChannel),
				ecalcond::CondExtractor(ecalcond::extractSuperModules),
				ecalcond::CondExtractor(ecalcond::extractBarrel),
				ecalcond::CondExtractor(ecalcond::extractEndcap),
				ecalcond::CondExtractor(ecalcond::extractAll)
			};
			return fun[how];
		}


		typedef ecalcond::Container Class;
		typedef ExtractWhat<Class> What;
		static What what() { return What();}

		ValueExtractor(){}
		ValueExtractor(What const & what)
			: m_what(what)
		{
			// here one can make stuff really complicated... 
			// ask to make average on selected channels...
		}

		void compute(Class const & it){
			std::vector<float> res;
			extractor(m_what.how())(it,m_what.which(),res);
			swap(res);
		}

	private:
		What  m_what;

	};


	template<>
	std::string
		PayLoadInspector<EcalChannelStatus>::dump() const {
			std::stringstream ss;
			EcalCondHeader h;
			ss << EcalChannelStatusXMLTranslator::dumpXML(h,object());
			return ss.str();

	}

	std::string StCodeToStr(unsigned int const & statusCode){
		unsigned int bits_1to5, bits, mask;
		mask = 0x1F;//get first 5 bits;
		bits_1to5 = mask & statusCode;
		std::stringstream ss;
		switch(bits_1to5){
			case 0: break;
			case 1: ss<< "["<< bits_1to5 << "]:" << "DAC settings problem, pedestal not in the design range, " << "Reco Action: standard reco."; break;
			case 2: ss<< "["<< bits_1to5 << "]:" << "channel with no laser, ok elsewhere; " << "Reco Action: standard reco."; break;
			case 3: ss<< "["<< bits_1to5 << "]:" << "noisy; " << "Reco Action: standard reco."; break;
			case 4: ss<< "["<< bits_1to5 << "]:" << "very noisy; " << "Reco Action: special reco (to be decided)."; break;
			case 5: ss<< "["<< bits_1to5 << "]: reserved for more categories of noisy channels."; break;
			case 6: ss<< "["<< bits_1to5 << "]: reserved for more categories of noisy channels."; break;
			case 7: ss<< "["<< bits_1to5 << "]: reserved for more categories of noisy channels."; break;
			case 8: ss<< "["<< bits_1to5 << "]:" << " channel at fixed gain 6 (or 6 and 1); " << "Reco Action: 3+5 or 3+1, special reco."; break;
			case 9: ss<< "["<< bits_1to5 << "]:" << " channel at fixed gain 1; " << "Reco Action: 3+5 or 3+1, special reco."; break;
			case 10: ss<< "["<< bits_1to5 << "]:" << " channel at fixed gain 0 (dead of type this); " << "Reco Action: recovery from neighbours."; break;
			case 11: ss<< "["<< bits_1to5 << "]:" << " non responding isolated channel (dead of type other); " << "Reco Action: recovery from neighbours."; break;
			case 12: ss<< "["<< bits_1to5 << "]:" << " channel and one or more neigbors not responding (e.g.: in a dead VFE 5x1 channel) " << "Reco Action: no recovery."; break;
			case 13: ss<< "["<< bits_1to5 << "]:" << " channel in TT with no data link, TP data ok; " << "Reco Action: recovery from TP data."; break;
			case 14: ss<< "["<< bits_1to5 << "]:" << " channel in TT with no data link and no TP data; " << "Reco Action: none."; break;
			default:
				ss<< "no such bits can be set!";
		}
		bits = 6;

		mask = 1 << bits;
		if (statusCode & mask) //case 15
			ss << " bit["<< bits << "]:" << " HV good/not good;";

		mask = 1 << ++bits;
		if (statusCode & mask)//case 16
			ss << " bit["<< bits << "]:" << " LV on/off;";

		mask = 1 << ++bits;
		if (statusCode & mask)//case 17
			ss << " bit["<< bits << "]:" << " DAQ in/out;";

		mask = 1 << ++bits;
		if (statusCode & mask)//case 18
			ss << " bit["<< bits << "]:" << " TP readout on/off;";

		mask = 1 << ++bits;
		if (statusCode & mask)//case 19
			ss << " bit["<< bits << "]:" << " Trigger in/out;";

		mask = 1 << ++bits;
		if (statusCode & mask)//case 20
			ss << " bit["<< bits << "]:" << " Temperature ok/not ok;";

		mask = 1 << ++bits;
		if (statusCode & mask)//case 21
			ss << " bit["<< bits << "]:" << " Humidity ok/not;";

		//for debugging:
		//ss << " StatusCode: " << statusCode << " bits_1to5: " <<bits_1to5 << std::endl;
		ss << std::endl;
		return ss.str();
	}

	void setErr(unsigned int const & statusCode,  std::vector< std::pair< std::string, unsigned int> > & vTotalWithErr){
		unsigned int bits_1to5, bits, mask;
		mask = 0x1F;//get first 5 bits;

		bits_1to5 = mask & statusCode;
		if ((bits_1to5 > 0) &&( bits_1to5 < 15)){
			vTotalWithErr[bits_1to5].second += 1;
		}

		bits = 6;

		mask = 1 << bits;
		if (statusCode & mask) //case 15
			vTotalWithErr[bits + 11].second += 1;

		mask = 1 << ++bits;
		if (statusCode & mask)//case 16
			vTotalWithErr[bits + 11].second += 1;

		mask = 1 << ++bits;
		if (statusCode & mask)//case 17
			vTotalWithErr[bits + 11].second += 1;

		mask = 1 << ++bits;
		if (statusCode & mask)//case 18
			vTotalWithErr[bits + 11].second += 1;

		mask = 1 << ++bits;
		if (statusCode & mask)//case 19
			vTotalWithErr[bits + 11].second += 1;

		mask = 1 << ++bits;
		if (statusCode & mask)//case 20
			vTotalWithErr[bits + 11].second += 1;

		mask = 1 << ++bits;
		if (statusCode & mask)//case 21
			vTotalWithErr[bits + 11].second += 1;
	}

	std::string getTotalErrors(const std::vector<EcalChannelStatusCode> & vItems){
		unsigned int stcode = 0, index = 0, totalErrorCodes = 22;
		std::stringstream ss;
		std::vector< std::pair< std::string, unsigned int > > vTotalWithErr(totalErrorCodes);

		
		// init error codes:
		index = 0;
		ss.str(""); ss << " No errors. "; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << "DAC settings problem, pedestal not in the design range, " << "Reco Action: standard reco."; vTotalWithErr[index].first = ss.str(); ++index;//1
		ss.str(""); ss << "["<< index << "]:" << "channel with no laser, ok elsewhere; " << "Reco Action: standard reco."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << "noisy; " << "Reco Action: standard reco."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << "very noisy; " << "Reco Action: special reco (to be decided)."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]: reserved for more categories of noisy channels."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]: reserved for more categories of noisy channels."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]: reserved for more categories of noisy channels."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << " channel at fixed gain 6 (or 6 and 1); " << "Reco Action: 3+5 or 3+1, special reco."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << " channel at fixed gain 1; " << "Reco Action: 3+5 or 3+1, special reco."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << " channel at fixed gain 0 (dead of type this); " << "Reco Action: recovery from neighbours."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << " non responding isolated channel (dead of type other); " << "Reco Action: recovery from neighbours."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << " channel and one or more neigbors not responding (e.g.: in a dead VFE 5x1 channel) " << "Reco Action: no recovery."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << " channel in TT with no data link, TP data ok; " << "Reco Action: recovery from TP data."; vTotalWithErr[index].first = ss.str(); ++index;
		ss.str(""); ss << "["<< index << "]:" << " channel in TT with no data link and no TP data; " << "Reco Action: none."; vTotalWithErr[index].first = ss.str(); ++index;

			unsigned int bits = 6;
			ss.str(""); ss << " bit["<< bits++ << "]:" << " HV good/not good;"; vTotalWithErr[index].first = ss.str(); ++index;//15
			ss.str(""); ss << " bit["<<  bits++ << "]:" << " LV on/off;"; vTotalWithErr[index].first = ss.str(); ++index;
			ss.str(""); ss << " bit["<<  bits++ << "]:" << " DAQ in/out;"; vTotalWithErr[index].first = ss.str(); ++index;
			ss.str(""); ss << " bit["<<  bits++ << "]:" << " TP readout on/off;"; vTotalWithErr[index].first = ss.str(); ++index;
			ss.str(""); ss << " bit["<<  bits++ << "]:" << " Trigger in/out;"; vTotalWithErr[index].first = ss.str(); ++index;
			ss.str(""); ss << " bit["<<  bits++ << "]:" << " Temperature ok/not ok;"; vTotalWithErr[index].first = ss.str(); ++index;
			ss.str(""); ss << " bit["<<  bits++ << "]:" << " Humidity ok/not;"; vTotalWithErr[index].first = ss.str(); ++index;//21
		
		
		ss.str("");
		//count error codes for each error:
		for (std::vector<EcalChannelStatusCode>::const_iterator vIter = vItems.begin(); vIter != vItems.end(); ++vIter){
			stcode = vIter->getStatusCode();
			if (stcode !=0){
				setErr(stcode, vTotalWithErr);
			}
		}

		unsigned int i = 0;
		for (std::vector< std::pair< std::string, unsigned int > >::const_iterator iTotalErr = vTotalWithErr.begin(); iTotalErr != vTotalWithErr.end(); ++iTotalErr){
			if (iTotalErr->second != 0)
				ss << "Total:" << iTotalErr->second << " Error:" << iTotalErr->first << std::endl;
			++i;
		}
		return ss.str();
	}

	template<>
	std::string PayLoadInspector<EcalChannelStatus>::summary() const {
		const std::vector<EcalChannelStatusCode> barrelItems = object().barrelItems();
		const std::vector<EcalChannelStatusCode> endcapItems = object().endcapItems();
		std::stringstream ss;

		ss << std::endl << "------Barrel Items: " << std::endl;
		ss << "---Total: " << barrelItems.size() << std::endl;
		ss << "---With errors: " << ecalcond::bad(barrelItems) << std::endl;
		ss << getTotalErrors(barrelItems);

		ss << std::endl << "------Endcap Items: " << std::endl;
		ss << "---Total: " << endcapItems.size() << std::endl;
		ss << "---With errors: " << ecalcond::bad(endcapItems) << std::endl;
		ss << getTotalErrors(endcapItems);

		//for (std::vector<EcalChannelStatusCode>::const_iterator barIter = barrelItems.begin(); barIter != barrelItems.end(); ++barIter){
		//	stcode = barIter->getStatusCode();
		//	if (stcode !=0){
		//		barrelId = EBDetId::detIdFromDenseIndex(count);
		//		ss << "Id[" << count << "] RawId:[" << barrelId.rawId() << "]" 
		//		 << "; Ieta:" << barrelId.ieta() <<
		//		"; Iphi:" << barrelId.iphi()<<
		//		"; Ism:" << barrelId.ism()<<
		//		"; Ic:" << barrelId.ic()<<
		//		"; Errors:" << StCodeToStr(barIter->getStatusCode(), vBarrWithErr, barrelId);
		//	}
		//	++count;
		//}

		//EEDetId endcapId;
		//ss << std::endl << "------Endcap Items: " << std::endl;
		//ss << "---Total: " << endcapItems.size() << std::endl;
		//ss << "---With errors: " << ecalcond::bad(endcapItems) << std::endl;
		//count = 0;
		//for (std::vector<EcalChannelStatusCode>::const_iterator endcapIter = endcapItems.begin(); endcapIter != endcapItems.end(); ++endcapIter){
		//	stcode = endcapIter->getStatusCode();
		//	if (stcode !=0){
		//		endcapId = EEDetId::detIdFromDenseIndex(count);
		//		ss << "Id[" << count << "] RawId:[" << endcapId.rawId() << "]" 
		//		 << "; Zside:" << endcapId.zside() <<
		//		"; Ix:" << endcapId.ix()<<
		//		"; Iy:" << endcapId.iy()<<
		//		"; Errors:" << StCodeToStr(endcapIter->getStatusCode());
		//	}
		//	++count;
		//}

		//ss << StCodeToStr(0xFFF09);

		return ss.str();
	}


	template<>
	std::string PayLoadInspector<EcalChannelStatus>::plot(std::string const & filename,
		std::string const &, 
		std::vector<int> const&, 
		std::vector<float> const& ) const {
			// use David's palette
			gStyle->SetPalette(1);

			const Int_t NRGBs = 5;
			const Int_t NCont = 255;

			Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
			Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
			Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
			Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
			TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
			gStyle->SetNumberContours(NCont);

			TCanvas canvas("CC map","CC map",800,800);
			TPad* padb = new TPad("padb","padb", 0., 0.55, 1., 1.);
			padb->Draw();
			TPad* padem = new TPad("padem","padem", 0., 0., 0.45, 0.45);
			padem->Draw();
			TPad* padep = new TPad("padep","padep", 0.55, 0., 1., 0.45);
			padep->Draw();

			const int kSides       = 2;
			const int kBarlRings   = EBDetId::MAX_IETA;
			const int kBarlWedges  = EBDetId::MAX_IPHI;
			const int kEndcWedgesX = EEDetId::IX_MAX;
			const int kEndcWedgesY = EEDetId::IY_MAX;

			TH2F* barrel = new TH2F("EB","EB Channel Status",360,0,360, 171, -85,86);
			TH2F* endc_p = new TH2F("EE+","EE+ Channel Status",100,1,101,100,1,101);
			TH2F* endc_m = new TH2F("EE-","EE- Channel Status",100,1,101,100,1,101);

			for (int sign=0; sign < kSides; sign++) {
				int thesign = sign==1 ? 1:-1;

				for (int ieta=0; ieta<kBarlRings; ieta++) {
					for (int iphi=0; iphi<kBarlWedges; iphi++) {
						EBDetId id((ieta+1)*thesign, iphi+1);
						barrel->Fill(iphi, ieta*thesign + thesign, object()[id.rawId()].getStatusCode());
						//	  if(iphi < 20 && object()[id.rawId()].getStatusCode() > 10)
						//	    std::cout << " phi " << iphi << " eta " << ieta << " status " << object()[id.rawId()].getStatusCode() << std::endl;
					}  // iphi
				}   // ieta

				for (int ix=0; ix<kEndcWedgesX; ix++) {
					for (int iy=0; iy<kEndcWedgesY; iy++) {
						if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
						EEDetId id(ix+1,iy+1,thesign);
						if (thesign==1) {
							endc_p->Fill(ix+1,iy+1,object()[id.rawId()].getStatusCode());
						}
						else{ 
							endc_m->Fill(ix+1,iy+1,object()[id.rawId()].getStatusCode());
						}
					}  // iy
				}   // ix
			}    // side

			TLine* l = new TLine(0., 0., 0., 0.);
			l->SetLineWidth(1);
			padb->cd();
			barrel->SetStats(0);
			barrel->SetMaximum(14);
			barrel->SetMinimum(0);
			barrel->Draw("colz");
			for(int i = 0; i <17; i++) {
				Double_t x = 20.+ (i *20);
				l = new TLine(x,-85.,x,86.);
				l->Draw();
			}
			l = new TLine(0.,0.,360.,0.);
			l->Draw();
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

					padem->cd();
					endc_m->SetStats(0);
					endc_m->SetMaximum(14);
					endc_m->SetMinimum(0);
					endc_m->Draw("colz");
					for ( int i=0; i<201; i=i+1) {
						if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
							(ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
								l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
									ixSectorsEE[i+1], iySectorsEE[i+1]);
						}
					}
					padep->cd();
					endc_p->SetStats(0);
					endc_p->SetMaximum(14);
					endc_p->SetMinimum(0);
					endc_p->Draw("colz");
					for ( int i=0; i<201; i=i+1) {
						if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
							(ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
								l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
									ixSectorsEE[i+1], iySectorsEE[i+1]);
						}
					}
					canvas.SaveAs(filename.c_str());
					return filename;
	}
}

namespace condPython {
	template<>
	void defineWhat<cond::ecalcond::Container>() {
		using namespace boost::python;
		enum_<cond::ecalcond::How>("How")
			.value("singleChannel",cond::ecalcond::singleChannel)
			.value("bySuperModule",cond::ecalcond::bySuperModule) 
			.value("barrel",cond::ecalcond::barrel)
			.value("endcap",cond::ecalcond::endcap)
			.value("all",cond::ecalcond::all)
			;

		typedef cond::ExtractWhat<cond::ecalcond::Container> What;
		class_<What>("What",init<>())
			.def("set_how",&What::set_how)
			.def("set_which",&What::set_which)
			.def("how",&What::how, return_value_policy<copy_const_reference>())
			.def("which",&What::which, return_value_policy<copy_const_reference>())
			;
	}
}

PYTHON_WRAPPER(EcalChannelStatus,EcalChannelStatus);
