// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorTowerProducer
// 

/**\class CastorTowerProducer CastorTowerProducer.cc RecoLocalCalo/Castor/src/CastorTowerProducer.cc

 Description: CastorTower Reconstruction Producer. Produce CastorTowers from CastorCells.
 Implementation:
*/

//
// Original Author:  Hans Van Haevermaet, Benoit Roland
//         Created:  Wed Jul  9 14:00:40 CEST 2008
// $Id: CastorTowerProducer.cc,v 1.4 2010/01/25 13:35:12 vlimant Exp $
//
//


// system include 
#include <memory>
#include <vector>
#include <iostream>
#include <TMath.h>
#include <TRandom3.h>

// user include 
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Point3D.h"

// Castor Object include
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"

//
// class declaration
//

class CastorTowerProducer : public edm::EDProducer {
   public:
      explicit CastorTowerProducer(const edm::ParameterSet&);
      ~CastorTowerProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void ComputeTowerVariable(const reco::CastorCellRefVector& usedCells, double&  Ehot, double& depth);
      
      // member data
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef ROOT::Math::RhoZPhiPoint CellPoint;
      typedef std::vector<reco::CastorCell> CastorCellCollection;
      typedef std::vector<reco::CastorTower> CastorTowerCollection;
      typedef edm::RefVector<CastorCellCollection>  CastorCellRefVector;
      std::string input_;
      double towercut_;
};

//
// constants, enums and typedefs
//

const double MYR2D = 180/M_PI;

//
// static data member definitions
//

//
// constructor and destructor
//

CastorTowerProducer::CastorTowerProducer(const edm::ParameterSet& iConfig) :
  input_(iConfig.getUntrackedParameter<std::string>("inputprocess","CastorCellReco")),
  towercut_(iConfig.getUntrackedParameter<double>("towercut",1.))
{
  //register your products
  produces<CastorTowerCollection>();
  //now do what ever other initialization is needed
}


CastorTowerProducer::~CastorTowerProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void CastorTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;
  using namespace reco;
  using namespace std;
  using namespace TMath;
  
  // Produce CastorTowers from CastorCells
  
  edm::Handle<CastorCellCollection> InputCells;
  iEvent.getByLabel(input_,InputCells);

  auto_ptr<CastorTowerCollection> OutputTowers (new CastorTowerCollection);
   
  // get and check input size
  int nCells = InputCells->size();

  LogDebug("CastorTowerProducer")
    <<"2. entering CastorTowerProducer"<<endl;

  if (nCells==0)
    LogDebug("CastorTowerProducer") <<"Warning: You are trying to run the Tower algorithm with 0 input cells.";
  
  // declare castor array
  // (0,x): Energies - (1,x): emEnergies - (2,x): hadEnergies - (3,x): phi position
  
  double poscastortowerarray[4][16]; 
  double negcastortowerarray[4][16];

  CastorCellRefVector poscastorusedcells[16];
  CastorCellRefVector negcastorusedcells[16];

  // set phi values and everything else to zero
  for (int j = 0; j < 16; j++) {
    poscastortowerarray[3][j] = -2.94524 + j*0.3927;
    poscastortowerarray[0][j] = 0.;
    poscastortowerarray[1][j] = 0.;
    poscastortowerarray[2][j] = 0.;

    negcastortowerarray[3][j] = -2.94524 + j*0.3927;
    negcastortowerarray[0][j] = 0.;
    negcastortowerarray[1][j] = 0.;
    negcastortowerarray[2][j] = 0.;
  }

  // loop over cells to build castortowerarray[4][16] and castorusedcells[16] 
  for (unsigned int i = 0; i < InputCells->size(); i++) {
    
    reco::CastorCellRef cell_p = reco::CastorCellRef(InputCells, i);
    
    double Ecell = cell_p->energy();
    double zcell = cell_p->z();
    double phicell = cell_p->phi();

    // loop over the 16 towers possibilities
    for ( int j=0;j<16;j++) {
      
      // phi matching condition
      if (TMath::Abs(phicell - poscastortowerarray[3][j]) < 0.0001) {

	// condition over cell z value
    	if (zcell > 0.) {
	  poscastortowerarray[0][j]+=Ecell;
	  if (TMath::Abs(zcell) < 14488) poscastortowerarray[1][j]+=Ecell;  
	  else poscastortowerarray[2][j]+=Ecell;
	  poscastorusedcells[j].push_back(cell_p);
	}
      
	else {
	  negcastortowerarray[0][j]+=Ecell;
	  if (TMath::Abs(zcell) < 14488) negcastortowerarray[1][j]+=Ecell;  
	  else negcastortowerarray[2][j]+=Ecell;
	  negcastorusedcells[j].push_back(cell_p);
	} // end condition over cell z value

      } // end phi matching condition
    } // end loop over the 16 towers possibilities
  } // end loop over cells to build castortowerarray[4][16] and castorusedcells[16]
  
  // make towers of the arrays

  double fem, Ehot, depth;
  double rhoTower = 88.5;

  // loop over the 16 towers possibilities
  for (int k=0;k<16;k++) {
    
    fem = 0;
    Ehot = 0;
    depth = 0;

    // select the positive towers with E > Ecut
    if (poscastortowerarray[0][k] > towercut_) {
      
      fem = poscastortowerarray[1][k]/poscastortowerarray[0][k];
      CastorCellRefVector usedCells = poscastorusedcells[k];
      ComputeTowerVariable(usedCells,Ehot,depth);

      LogDebug("CastorTowerProducer")
	<<"tower "<<k+1<<": fem = "<<fem<<" ,depth = "<<depth<<" ,Ehot = "<<Ehot<<endl;

      TowerPoint temptowerposition(rhoTower,5.9,poscastortowerarray[3][k]);
      Point towerposition(temptowerposition);

      CastorTower newtower(poscastortowerarray[0][k],towerposition,poscastortowerarray[1][k],poscastortowerarray[2][k],fem,depth,Ehot,
			   poscastorusedcells[k]);
      OutputTowers->push_back(newtower);
    } // end select the positive towers with E > Ecut
    
    // select the negative towers with E > Ecut
    if (negcastortowerarray[0][k] > towercut_) {
      
      fem = negcastortowerarray[1][k]/negcastortowerarray[0][k];
      CastorCellRefVector usedCells = negcastorusedcells[k];
      ComputeTowerVariable(usedCells,Ehot,depth);

      LogDebug("CastorTowerProducer")
	<<"tower "<<k+1<<": fem = "<<fem<<" ,depth = "<<depth<<" ,Ehot = "<<Ehot<<endl;

      TowerPoint temptowerposition(rhoTower,-5.9,negcastortowerarray[3][k]);
      Point towerposition(temptowerposition);

      CastorTower newtower(negcastortowerarray[0][k],towerposition,negcastortowerarray[1][k],negcastortowerarray[2][k],fem,depth,Ehot,
			   negcastorusedcells[k]);
      OutputTowers->push_back(newtower);
    } // end select the negative towers with E > Ecut
    
  } // end loop over the 16 towers possibilities
  
  iEvent.put(OutputTowers);
} 


// ------------ method called once each job just before starting event loop  ------------
void CastorTowerProducer::beginJob() {
  LogDebug("CastorTowerProducer")
    <<"Starting CastorTowerProducer";
}

// ------------ method called once each job just after ending the event loop  ------------
void CastorTowerProducer::endJob() {
  LogDebug("CastorTowerProducer")
    <<"Ending CastorTowerProducer";
}

void CastorTowerProducer::ComputeTowerVariable(const reco::CastorCellRefVector& usedCells, double&  Ehot, double& depth) {

  using namespace reco;
  using namespace std;

  double Etot = 0;

  // loop over the cells used in the tower k
  for (CastorCell_iterator it = usedCells.begin(); it != usedCells.end(); it++) {
    reco::CastorCellRef cell_p = *it;

    double Ecell = cell_p->energy();
    double zcell = cell_p->z();

    if(Ecell > Ehot) Ehot = Ecell;
    depth+=Ecell*zcell;
    Etot+=Ecell;
  }

  depth/=Etot;
  Ehot/=Etot;
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorTowerProducer);
