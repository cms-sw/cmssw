#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TMultiGraph.h"
#include "TLegend.h"

#include <string>
#include <sstream>

namespace cond {

	template<>
	class ValueExtractor<BeamSpotObjects>: public  BaseValueExtractor<BeamSpotObjects> {
	public:
		typedef BeamSpotObjects Class;
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
	std::string PayLoadInspector<BeamSpotObjects>::summary() const {
		std::stringstream ss;
		object().print(ss);
		return ss.str();
	}

	template<>
	std::string PayLoadInspector<BeamSpotObjects>::plot(std::string const & filename,
		std::string const &,
		std::vector<int> const &,
		std::vector<float> const& ) const 
	{

		TCanvas canvas("iC","iC",800,800);    

		canvas.SaveAs(filename.c_str());
		return filename;
	}

	template<>
	std::string PayLoadInspector<BeamSpotObjects>::trend_plot(std::string const & filename,
		std::string const & opt_string,
		std::vector<int> const& ints,
		std::vector<float> const & floats,
		std::vector<std::string> const& strings) const 
	{
		std::stringstream ss("");

		if (strings.size() < 2)
			return ("Error! Not enough data for initializing connection for making plots! (from \
					template<>\
					std::string PayLoadInspector<BeamSpotObjects>::trend_plot)");

		std::vector<std::string>::const_iterator iter_beg = strings.begin();

		std::string conString= (*iter_beg); 
		++iter_beg;
		std::string authPath = (*iter_beg);
		++iter_beg;

		//make connection object
		DbConnection dbConn;

		//set in configuration object authentication path
		dbConn.configuration().setAuthenticationPath(authPath);
		dbConn.configure();

		//create session object from connection
		DbSession dbSes=dbConn.createSession();

		//try to make connection
		dbSes.open(conString,true);

		//start a transaction (true=readOnly)
		dbSes.transaction().start(true);

		//get the actual object
		boost::shared_ptr<BeamSpotObjects> ptrBeamSpot;

		//iter_beg now stands on first token in the vector
		std::string token;

		std::vector<float> vecX;
		std::vector<float> vecY;
		std::vector<float> vecZ;
		for (std::vector<float>::const_iterator iter_float = floats.begin() ;iter_beg != strings.end();++iter_beg, ++iter_float){
			token = (*iter_beg);
			//std::cout << token << " ";
			
			ptrBeamSpot = dbSes.getTypedObject<BeamSpotObjects>(token);

			//get data from the objects:
			vecX.push_back(ptrBeamSpot->GetX());
			vecY.push_back(ptrBeamSpot->GetY());
			vecZ.push_back(ptrBeamSpot->GetZ());
			std::cout << "since: "<<(*iter_float)<< " X: "<< ptrBeamSpot->GetX() << " Y: "<< ptrBeamSpot->GetY() << " Z: " << ptrBeamSpot->GetZ() << std::endl;


		}
		
		//close db session
		dbSes.close();

		TCanvas canvas("iC","iC",1200,1200); 
		//canvas.UseCurrentStyle();
		//gStyle->SetPalette(1);
		std::cout << *(floats.end() -1) << "   " << *(floats.begin());
		float max = *(floats.end() -1);
		float min =  *(floats.begin());

		unsigned int lineWidth = 2;
		unsigned int startColor = 2;

		float result = ((max - min) / max) ;
		if ( result >= 0.1 ){
			canvas.SetLogx(1);
		}
		std::cout << "result: " << result << std::endl;

	
	//1)
		TGraph graphdataX(vecX.size(), static_cast<const float *>(&floats[0]), static_cast<const float *>(&vecX[0]));
		//graphdataX.GetXaxis()->SetRangeUser(*(floats.begin()),*(floats.end() -1));
		//graphdataX.GetXaxis()-> SetLabelSize(0.03);
		graphdataX.SetLineColor(startColor++);
		graphdataX.SetLineWidth(lineWidth);

	//2)
		TGraph graphdataY(vecY.size(), static_cast<const float *>(&floats[0]), static_cast<const float *>(&vecY[0]));
		//graphdataY.GetXaxis()->SetRangeUser(*(floats.begin()),*(floats.end() -1));
		//graphdataY.GetXaxis()-> SetLabelSize(0.03);
		graphdataY.SetLineColor(startColor++);
		graphdataY.SetLineWidth(lineWidth);

	//3)
		TGraph graphdataZ(vecZ.size(), static_cast<const float *>(&floats[0]), static_cast<const float *>(&vecZ[0]));
		//graphdataZ.GetXaxis()->SetRangeUser(*(floats.begin()),*(floats.end() -1));
		//graphdataZ.GetXaxis()-> SetLabelSize(0.03);
		graphdataZ.SetLineColor(startColor++);
		graphdataZ.SetLineWidth(lineWidth);


		TMultiGraph mg;
		std::stringstream plotTitle;
		plotTitle.precision(0);
		plotTitle << std::fixed;
		//plotTitle << "BeamSpot trend graph. X0 = black, Y0 = red, Z0 = green; since(first="
		plotTitle << "BeamSpot trend graph (first="
					<<(double)min 
			<< ", last=" 
			<< (double) max 
			<< ", total="
			<< floats.size()
			<<");since;value";
		mg.SetTitle( plotTitle.str().c_str());

		//graphdataX.GetXaxis()->SetBinLabel(2,"mylabel");
		//graphdataY.GetXaxis()->SetBinLabel(2,"mylabel");
		//graphdataZ.GetXaxis()->SetBinLabel(2,"mylabel");

		mg.Add(&graphdataX);
		mg.Add(&graphdataY);
		mg.Add(&graphdataZ);
		

		//float size1 = mg.GetXaxis()-> GetBinWidth();
		//std::cout <<" BinWidth: " << size1 << std::endl;

		mg.Draw("LA*");

		TLegend leg(0.7,0.7,0.9,0.9,"Beam Spot Legend");
		leg.AddEntry(&graphdataX,"X_{0}","lpf");
		leg.AddEntry(&graphdataY,"Y_{0}","lpf");
		leg.AddEntry(&graphdataZ,"Z_{0}","lpf");

		leg.Draw();

		ss.str("");
		ss << filename << ".png";
		canvas.SaveAs((ss.str()).c_str());
		return ss.str();
	}
}


PYTHON_WRAPPER(BeamSpotObjects,BeamSpotObjects);



