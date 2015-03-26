#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondTools/Ecal/interface/EcalPulseShapesXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

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

#include "CondCore/EcalPlugins/plugins/EcalPyWrapperFunctions.h"

namespace {
	struct Printer {
		void doit(EcalPulseShape const & item) {
                  for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s)
                    ss << item.val(s);
			ss << " ";
		}
		std::stringstream ss;
	};
}

namespace cond {

	namespace ecalpulseshape {
          enum Quantity { sample_0=1, sample_1=2, sample_2=3, sample_3=4, sample_4=5, sample_5=6, sample_6=7, sample_7=8, sample_8=9, sample_9=10, sample_10=11, sample_11=12 };
		enum How { singleChannel, bySuperModule, all};

		float average(EcalPulseShapes const & pulseshapes, Quantity q) {
			return std::accumulate(
				boost::make_transform_iterator(pulseshapes.barrelItems().begin(),bind(&EcalPulseShape::val,_1,q-1)),
				boost::make_transform_iterator(pulseshapes.barrelItems().end(),bind(&EcalPulseShape::val,_1,q-1)),
				0.)/float(pulseshapes.barrelItems().size());
		}

		void extractAverage(EcalPulseShapes const & pulseshapes, Quantity q, std::vector<int> const &,  std::vector<float> & result) {
			result.resize(1);
			result[0] = average(pulseshapes,q);
		}

		void extractSuperModules(EcalPulseShapes const & pulseshapes, Quantity q, std::vector<int> const & which,  std::vector<float> & result) {
			// bho...
		}

		void extractSingleChannel(EcalPulseShapes const & pulseshapes, Quantity q, std::vector<int> const & which,  std::vector<float> & result) {
			for (unsigned int i=0; i<which.size();i++) {
				// absolutely arbitraty
				if ((unsigned int) (which[i])<  pulseshapes.barrelItems().size())
					result.push_back( pulseshapes.barrelItems()[which[i]].val(q-1));
			}
		}

		typedef boost::function<void(EcalPulseShapes const & pulseshapes, Quantity q, std::vector<int> const & which,  std::vector<float> & result)> PulseShapeExtractor;
	}

	template<>
	struct ExtractWhat<EcalPulseShapes> {

		ecalpulseshape::Quantity m_quantity;
		ecalpulseshape::How m_how;
		std::vector<int> m_which;

		ecalpulseshape::Quantity const & quantity() const { return m_quantity;}
		ecalpulseshape::How const & how() const { return m_how;}
		std::vector<int> const & which() const { return m_which;}


		void set_quantity( ecalpulseshape::Quantity i) { m_quantity=i;}
		void set_how(ecalpulseshape::How i) {m_how=i;}
		void set_which(std::vector<int> & i) { m_which.swap(i);}
	};


	template<>
	class ValueExtractor<EcalPulseShapes>: public  BaseValueExtractor<EcalPulseShapes> {
	public:

		static ecalpulseshape::PulseShapeExtractor & extractor(ecalpulseshape::How how) {
			static  ecalpulseshape::PulseShapeExtractor fun[3] = { 
				ecalpulseshape::PulseShapeExtractor(ecalpulseshape::extractSingleChannel),
				ecalpulseshape::PulseShapeExtractor(ecalpulseshape::extractSuperModules),
				ecalpulseshape::PulseShapeExtractor(ecalpulseshape::extractAverage)
			};
			return fun[how];
		}

		typedef EcalPulseShapes Class;
		typedef ExtractWhat<Class> What;
		static What what() { return What();}

		ValueExtractor(){}
		ValueExtractor(What const & what)
			: m_what(what)
		{
			// here one can make stuff really complicated... (select mean rms, 12,6,1)
			// ask to make average on selected channels...
		}

		void compute(Class const & it) override{
			std::vector<float> res;
			extractor(m_what.how())(it,m_what.quantity(),m_what.which(),res);
			swap(res);
		}

	private:
		What  m_what;

	};


	template<>
	std::string
		PayLoadInspector<EcalPulseShapes>::dump() const {
			std::stringstream ss;
			EcalCondHeader header;
			ss<<EcalPulseShapesXMLTranslator::dumpXML(header,object());
			return ss.str();
	}  // dump


	class EcalPulseShapesHelper: public EcalPyWrapperHelper<EcalPulseShape>{
	public:
          EcalPulseShapesHelper():EcalPyWrapperHelper<EcalObject>(EcalPulseShape::TEMPLATESAMPLES){}
	protected:
		typedef EcalPulseShape EcalObject;
		type_vValues getValues( const std::vector<EcalPulseShape> & vItems) override
		{
			type_vValues vValues(total_values);

                        for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) vValues[s].first = Form("sample_%d",s);

                        //get info:
                        for(std::vector<EcalPulseShape>::const_iterator iItems = vItems.begin(); iItems != vItems.end(); ++iItems) {
                          for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) vValues[s].second = iItems->val(s);
                        }
                        return vValues;
		}
	};

	template<>
	std::string PayLoadInspector<EcalPulseShapes>::summary() const {
		std::stringstream ss;
		EcalPulseShapesHelper helper;
		ss << helper.printBarrelsEndcaps(object().barrelItems(), object().endcapItems());
		return ss.str();
	}  // summary


	// return the real name of the file including extension...
	template<>
	std::string PayLoadInspector<EcalPulseShapes>::plot(std::string const & filename,
		std::string const &, 
		std::vector<int> const&, 
		std::vector<float> const& ) const {
			gStyle->SetPalette(1);
			//    TCanvas canvas("CC map","CC map",840,600);
			TCanvas canvas("CC map","CC map",800,1200);

			float xmi[3] = {0.0 , 0.22, 0.78};
			float xma[3] = {0.22, 0.78, 1.00};
			TPad*** pad = new TPad**[6];
			for (int s = 0; s < 6; s++) {
				pad[s] = new TPad*[3];
				for (int obj = 0; obj < 3; obj++) {
					float yma = 1.- (0.17 * s);
					float ymi = yma - 0.15;
					pad[s][obj] = new TPad(Form("p_%i_%i", obj, s),Form("p_%i_%i", obj, s),
						xmi[obj], ymi, xma[obj], yma);
					pad[s][obj]->Draw();
				}
			}

			const int kSides       = 2;
			const int kBarlRings   = EBDetId::MAX_IETA;
			const int kBarlWedges  = EBDetId::MAX_IPHI;
			const int kEndcWedgesX = EEDetId::IX_MAX;
			const int kEndcWedgesY = EEDetId::IY_MAX;

			TH2F** barrel_s = new TH2F*[EcalPulseShape::TEMPLATESAMPLES];
			TH2F** endc_p_s = new TH2F*[EcalPulseShape::TEMPLATESAMPLES];
			TH2F** endc_m_s = new TH2F*[EcalPulseShape::TEMPLATESAMPLES];
                        for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) {
				barrel_s[s] = new TH2F(Form("EBs%i",s),Form("sample %i EB",s),360,0,360, 170, -85,85);
				endc_p_s[s] = new TH2F(Form("EE+s%i",s),Form("sample %i EE+",s),100,1,101,100,1,101);
				endc_m_s[s] = new TH2F(Form("EE-s%i",s),Form("sample %i EE-",s),100,1,101,100,1,101);
			}

			for (int sign=0; sign < kSides; sign++) {
				int thesign = sign==1 ? 1:-1;

				for (int ieta=0; ieta<kBarlRings; ieta++) {
					for (int iphi=0; iphi<kBarlWedges; iphi++) {
						EBDetId id((ieta+1)*thesign, iphi+1);
						float y = -1 - ieta;
						if(sign == 1) y = ieta;
                                                for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s) {
                                                  barrel_s[s]->Fill(iphi, y, object()[id.rawId()].pdfval[s]);
                                                }
					}  // iphi
				}   // ieta

				for (int ix=0; ix<kEndcWedgesX; ix++) {
					for (int iy=0; iy<kEndcWedgesY; iy++) {
						if (! EEDetId::validDetId(ix+1,iy+1,thesign)) continue;
						EEDetId id(ix+1,iy+1,thesign);
						if (thesign==1) {
                                                  for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s)
                                                    endc_p_s[s]->Fill(ix+1,iy+1,object()[id.rawId()].pdfval[s]);
						}
						else { 
                                                  for(int s = 0; s<EcalPulseShape::TEMPLATESAMPLES; ++s)
                                                    endc_m_s[s]->Fill(ix+1,iy+1,object()[id.rawId()].pdfval[s]);
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

                        int ipad=0;
                        // plot only the measured ones, not the extrapolated
                        for(int s = 0; s<7; ++s) {
                        // plot only the extrapolated ones
                        // for(int s = 7; s<12; ++s) {
                          // do not plot the maximum sample, which is 1 by default
                          if(s==2) continue;
                          pad[ipad][0]->cd();
                          endc_m_s[s]->SetStats(0);
                          endc_m_s[s]->SetMaximum(1);
                          endc_m_s[s]->SetMinimum(0);
                          endc_m_s[s]->Draw("colz");
                          for ( int i=0; i<201; i=i+1) {
                            if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
                                 (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
                              l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
                                          ixSectorsEE[i+1], iySectorsEE[i+1]);
                              l->SetLineWidth(0.2);
                            }
                          }
                          //canvas.cd(2);
                          pad[ipad][1]->cd();
                          barrel_s[s]->SetStats(0);
                          barrel_s[s]->SetMaximum(1);
                          barrel_s[s]->SetMinimum(0);
                          barrel_s[s]->Draw("colz");
                          for(int i = 0; i <17; i++) {
                            Double_t x = 20.+ (i *20);
                            l = new TLine(x,-85.,x,86.);
                            l->Draw();
                            l->SetLineWidth(0.2);
                          }
                          l = new TLine(0.,0.,360.,0.);
                          l->Draw();
                          //canvas.cd(3);
                          pad[ipad][2]->cd();
                          endc_p_s[s]->SetStats(0);
                          endc_p_s[s]->SetMaximum(1);
                          endc_p_s[s]->SetMinimum(0);
                          endc_p_s[s]->Draw("colz");
                          for ( int i=0; i<201; i=i+1) {
                            if ( (ixSectorsEE[i]!=0 || iySectorsEE[i]!=0) && 
                                 (ixSectorsEE[i+1]!=0 || iySectorsEE[i+1]!=0) ) {
                              l->DrawLine(ixSectorsEE[i], iySectorsEE[i], 
                                          ixSectorsEE[i+1], iySectorsEE[i+1]);
                              l->SetLineWidth(0.2);
                            }
                          }
                          ipad++;
                        }
                        
                        canvas.SaveAs(filename.c_str());
                        return filename;
	}  // plot
}

namespace condPython {
	template<>
	void defineWhat<EcalPulseShapes>() {
		using namespace boost::python;
		enum_<cond::ecalpulseshape::Quantity>("Quantity")
			.value("sample_0",cond::ecalpulseshape::sample_0)
			.value("sample_1",cond::ecalpulseshape::sample_1)
			.value("sample_2",cond::ecalpulseshape::sample_2)
			.value("sample_3",cond::ecalpulseshape::sample_3)
			.value("sample_4",cond::ecalpulseshape::sample_4)
			.value("sample_5",cond::ecalpulseshape::sample_5)
			.value("sample_6",cond::ecalpulseshape::sample_6)
			.value("sample_7",cond::ecalpulseshape::sample_7)
			.value("sample_8",cond::ecalpulseshape::sample_8)
			.value("sample_9",cond::ecalpulseshape::sample_9)
			;
		enum_<cond::ecalpulseshape::How>("How")
			.value("singleChannel",cond::ecalpulseshape::singleChannel)
			.value("bySuperModule",cond::ecalpulseshape::bySuperModule) 
			.value("all",cond::ecalpulseshape::all)
			;

		typedef cond::ExtractWhat<EcalPulseShapes> What;
		class_<What>("What",init<>())
			.def("set_quantity",&What::set_quantity)
			.def("set_how",&What::set_how)
			.def("set_which",&What::set_which)
			.def("quantity",&What::quantity, return_value_policy<copy_const_reference>())
			.def("how",&What::how, return_value_policy<copy_const_reference>())
			.def("which",&What::which, return_value_policy<copy_const_reference>())
			;
	}
}



PYTHON_WRAPPER(EcalPulseShapes,EcalPulseShapes);
