/** Derived from DTGeometryAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

class GEMGeometryAnalyzer : public edm::EDAnalyzer {

public: 
  GEMGeometryAnalyzer( const edm::ParameterSet& pset);

  ~GEMGeometryAnalyzer();

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
  const std::string& myName() { return myName_;}
  
private: 

  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::ofstream ofos;
};
using namespace std;
GEMGeometryAnalyzer::GEMGeometryAnalyzer( const edm::ParameterSet& /*iConfig*/ )
  : dashedLineWidth_(104), dashedLine_( string(dashedLineWidth_, '-') ), 
    myName_( "GEMGeometryAnalyzer" ) 
{ 
  ofos.open("HTMLtestOutput_27_04.html");
  ofos <<"<!DOCTYPE html>"<< endl;
  ofos <<"<html>"<< endl;
  ofos << "<head>" << endl << "<style>" << endl << "   table, th, td {" << endl << " border: 1px solid black;" << endl << " border-collapse: collapse;"<< endl << " }" << endl << " th {" << endl << "  padding: 5x;" << endl << "   text-align: center; " << endl << "color:red;" << endl << "font-size:160%;}" << endl <<  " td { " << endl << "  padding: 5px;" << endl <<"   text-align: center; "<< endl << " } " << endl << "</style>" << endl;
  ofos << "<script src=\"https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js\"></script>" << endl;
  ofos << "<script>" << endl
       << " $(document).ready(function(){" << endl
       << " $(\"#hide\").click(function(){" << endl
       << "  $(\"table\").hide();" << endl
       << "});" << endl
       << "     $(\"#show\").click(function(){" << endl
       << "  $(\"table\").show();" << endl
       << " });" << endl
       << " });" << endl
       << "</script>" << endl;
  ofos << "</head>" <<endl;
  ofos <<"<body>"<< endl; 
  ofos <<"<p1> ======================== Opening output file </p1>"<< endl;
}


GEMGeometryAnalyzer::~GEMGeometryAnalyzer() 
{
  ofos.close();
  ofos <<"======================== Closing output file"<< endl;
}

void GEMGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup )
{
  edm::ESHandle<GEMGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get(pDD);     
  
  ofos << myName() << ": Analyzer..." << endl;
  ofos << "start " << dashedLine_ << endl;

  ofos << " Geometry node for GEMGeom is  " << &(*pDD) << endl;   
  ofos << " detTypes       \t"              <<pDD->detTypes().size() << endl;
  ofos << " GeomDetUnit       \t"           <<pDD->detUnits().size() << endl;
  ofos << " GeomDet           \t"           <<pDD->dets().size() << endl;
  ofos << " GeomDetUnit DetIds\t"           <<pDD->detUnitIds().size() << endl;
  ofos << " eta partitions \t"              <<pDD->etaPartitions().size() << endl;
  ofos << " chambers       \t"              <<pDD->chambers().size() << endl;
  ofos << " super chambers  \t"             <<pDD->superChambers().size() << endl;
  ofos << " rings  \t\t"                    <<pDD->rings().size() << endl;
  ofos << " stations  \t\t"                 <<pDD->stations().size() << endl;
  ofos << " regions  \t\t"                  <<pDD->regions().size() << endl;

  // checking uniqueness of roll detIds 
  bool flagNonUniqueRollID = false;
  bool flagNonUniqueRollRawID = false;
  int nstrips = 0;
  int npads = 0;
  for (auto roll1 : pDD->etaPartitions()){
    nstrips += roll1->nstrips();
    npads += roll1->npads();
    for (auto roll2 : pDD->etaPartitions()){
      if (roll1 != roll2){
	if (roll1->id() == roll2->id()) flagNonUniqueRollID = true;
	if (roll1->id().rawId() == roll2->id().rawId()) flagNonUniqueRollRawID = true;
      }
    }
  }
  // checking the number of strips and pads
  ofos << " total number of strips\t"<<nstrips << endl;
  ofos << " total number of pads  \t"<<npads << endl;
  if (flagNonUniqueRollID or flagNonUniqueRollRawID)
    ofos << " -- WARNING: non unique roll Ids!!!" << endl;

  // checking uniqueness of chamber detIds
  bool flagNonUniqueChID = false;
  bool flagNonUniqueChRawID = false;
  for (auto ch1 : pDD->chambers()){
    for (auto ch2 : pDD->chambers()){
      if (ch1 != ch2){
	if (ch1->id() == ch2->id()) flagNonUniqueChID = true;
	if (ch1->id().rawId() == ch2->id().rawId()) flagNonUniqueChRawID = true;
      }
    }
  }
  if (flagNonUniqueChID or flagNonUniqueChRawID)
    ofos << " -- WARNING: non unique chamber Ids!!!" << endl;

  ofos << myName() << ": Begin iteration over geometry..." << endl;
  ofos << "iter " << dashedLine_ << endl;
  
  //----------------------- Global GEMGeometry TEST -------------------------------------------------------
  ofos << myName() << "Begin GEMGeometry structure TEST" << endl;
  ofos<<"<table style=\"width:100%\">" << endl << "<tr>" << endl;
  ofos << "<th> GEM Super Chamber Id </th>" << endl << "<th> Chamber </th>" << endl << "<th> Roll </th>" << endl << "<th> r (bottom) in cm </th>" << endl << "<th> W in cm </th>" << endl << "<th> h in cm </th>" << endl << "<th> cStrip1 phi </th>"<< endl << "<th> cStripN phi </th>"<< endl << "<th> dphi </th>"<< endl << "<th> global Z </th>"<< endl << "</tr>" << endl;
 
  for (auto region : pDD->regions()) {
    //  ofos << "  GEMRegion " << region->region() << " has " << region->nStations() << " stations." << endl;
    for (auto station : region->stations()) {
      // ofos << "    GEMStation " << station->getName() << " has " << station->nRings() << " rings." << endl;
      for (auto ring : station->rings()) {
	//	ofos << "      GEMRing " << ring->region() << " " << ring->station() << " " << ring->ring() << " has " << ring->nSuperChambers() << " super chambers." << endl;

	int i = 1;
	for (auto sch : ring->superChambers()) {
	  GEMDetId schId(sch->id());
	  //  if(i<19) ofos << "        GEMSuperChamber " << 2*i-1 << ", GEMDetId = " << schId.rawId() << ", " << schId << " has " << sch->nChambers() << " chambers." << endl;
	  //  else ofos << "        GEMSuperChamber " << 2*(i-18) << ", GEMDetId = " << schId.rawId() << ", " << schId << " has " << sch->nChambers() << " chambers." << endl;
	  ofos << "<tr>" << endl;
	  ofos << "<td>" << schId << "</td>" << endl;



	  // checking the dimensions of each partition & chamber
	  int j = 1;
	  for (auto ch : sch->chambers()){
	    GEMDetId chId(ch->id());
	      int nRolls(ch->nEtaPartitions());
	    // ofos << "          GEMChamber " << j << ", GEMDetId = " << chId.rawId() << ", " << chId << " has " << nRolls << " eta partitions." << endl;
	    ofos << "<td>" << j << "</td>" << endl;
	    //  else ofos << "<tr>" << endl << "<td> </td>" << endl << "<td>" << j << "</td>" << endl;
	    int k = 1;
	    auto& rolls(ch->etaPartitions());
	    //	     ofos << "<td>" << k << "</td>" << endl;
	    // else ofos << "<tr>" << endl<< "<td> </td>" << endl<< "<td> </td>" << endl << "<td>" << k << "</td>" << endl;
	    /*
	     * possible checklist for an eta partition:
	     *   base_bottom, base_top, height, strips, pads
	     *   cx, cy, cz, ceta, cphi
	     *   tx, ty, tz, teta, tphi
	     *   bx, by, bz, beta, bphi
	     *   pitch center, pitch bottom, pitch top
	     *   deta, dphi
	     *   gap thickness
	     *   sum of all dx + gap = chamber height
	     */      
	    
	    for (auto roll : rolls){
	      //  GEMDetId rId(roll->id());
	      //  ofos<<"            GEMEtaPartition " << k << ", GEMDetId = " << rId.rawId() << ", " << rId << endl;
	      ofos << "<td>" << k << "</td>" << endl;	      
	      const BoundPlane& bSurface(roll->surface());
	      const StripTopology* topology(&(roll->specificTopology()));
	      
	      // base_bottom, base_top, height, strips, pads (all half length)
	      auto& parameters(roll->specs()->parameters());
	      float bottomEdge(parameters[0]);
	      //  float topEdge(parameters[1]);
	      float height(parameters[2]);
	      float nStrips(parameters[3]);
	      //  float nPads(parameters[4]);
	      
	       LocalPoint  lCentre( 0., 0., 0. );
	      GlobalPoint gCentre(bSurface.toGlobal(lCentre));
	      /*
	      LocalPoint  lTop( 0., height, 0.);
	      GlobalPoint gTop(bSurface.toGlobal(lTop));
	      */
	      LocalPoint  lBottom( 0., -height, 0.);
	      GlobalPoint gBottom(bSurface.toGlobal(lBottom));
	      /*
	      //   gx, gy, gz, geta, gphi (center)
	      double cx(gCentre.x());
	      double cy(gCentre.y());
	      */
	      double cz(gCentre.z());
	      /*
	      double ceta(gCentre.eta());
	      int cphi(static_cast<int>(gCentre.phi().degrees()));
	      if (cphi < 0) cphi += 360;
	      
	      double tx(gTop.x());
	      double ty(gTop.y());
	      double tz(gTop.z());
	      double teta(gTop.eta());
	      int tphi(static_cast<int>(gTop.phi().degrees()));
	      if (tphi < 0) tphi += 360;
	      */
	      double bx(gBottom.x());
	      double by(gBottom.y());
	      //double bz(gBottom.z());
	      //double beta(gBottom.eta());
	      // int bphi(static_cast<int>(gBottom.phi().degrees()));
	      //if (bphi < 0) bphi += 360;
	      /* 
	      // pitch bottom, pitch top, pitch centre
	      float pitch(roll->pitch());
	      float topPitch(roll->localPitch(lTop));
	      float bottomPitch(roll->localPitch(lBottom));
	      
	      // Type - should be GHA[1-nRolls]
	      string type(roll->type().name());
	      */
	      // print info about edges
	      LocalPoint lEdge1(topology->localPosition(0.));
	      LocalPoint lEdgeN(topology->localPosition((float)nStrips));
	      
	      double cstrip1(roll->toGlobal(lEdge1).phi().degrees());
	      double cstripN(roll->toGlobal(lEdgeN).phi().degrees());
	      double dphi(fabs(cstripN - cstrip1));
	      if (dphi > 180.) dphi = (360.- dphi);

	      //double deta(abs(beta - teta));
	      const bool printDetails(true);
	      if (printDetails) {
		ofos //<< "|    \t\tType: " << type << endl
		  << "<td>" << std::sqrt(bx*bx + by*by) << "</td>" << endl << "<td>" << bottomEdge*2 << "</td>" << endl << "<td>" << height*2 << "</td> " << endl
		  //<< "|   \t\tnStrips = " << nStrips << ", nPads =  " << nPads << "*" << endl
		  // << "|    \t\ttop(x,y,z)[cm] = (" << tx << ", " << ty << ", " << tz << "), top(eta,phi) = (" << teta << ", " << tphi << ")*" << endl
		  // << "|    \t\tcenter(x,y,z)[cm] = (" << cx << ", " << cy << ", " << cz << "), center(eta,phi) = (" << ceta << ", " << cphi << ")*" << endl
		  // << "|    \t\tbottom(x,y,z)[cm] = (" << bx << ", " << by << ", " << bz << "), bottom(eta,phi) = (" << beta << ", " << bphi << ")*" << endl
		  // << "|    \t\tpitch (top,center,bottom) = (" << topPitch << ", " << pitch << ", " << bottomPitch << "),| dEta = " << deta 
		  << "<td>" << cstrip1 << " </td>" << endl<< "<td>" << cstripN << " </td>" << endl << "<td>" << dphi << "</td>" << endl << "<td>" << cz << "</td>" << endl << "</tr>" << endl;
	      }
	      if(k<=nRolls)ofos << "<tr>" << endl<< "<td> </td>" << endl<< "<td> </td>" << endl;
	      ++k;
	    }
	    if(j <= sch->nChambers()) ofos << "<tr>" << endl << "<td> </td>" << endl;
	    ++j;
	  }
	  ++i;
	}
      }
    }
  }
  ofos << "</table>" << endl;
  ofos << "<button id=\"hide\">Hide</button>" << endl;
  ofos << "<button id=\"show\">Show</button>" << endl;
  ofos << "<p>" <<  dashedLine_ << " end </p>" << endl << "</body>" << endl << "</html>" << endl;

}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(GEMGeometryAnalyzer);
