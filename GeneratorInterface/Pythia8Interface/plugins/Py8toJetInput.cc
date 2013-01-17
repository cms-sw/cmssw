#include "GeneratorInterface/Pythia8Interface/plugins/Py8toJetInput.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

const std::vector<fastjet::PseudoJet> 
Py8toJetInput::fillJetAlgoInput( const Event& event, const Event& workEvent,
                                 const lhef::LHEEvent* lhee,
                                 const std::vector<int>* typeIdx )
{

   fJetInput.clear();
   
   Event workEventJet = workEvent;
   
   const lhef::HEPEUP& hepeup = *lhee->getHEPEUP();
         
   std::set< int > typeSet[3];
   
   // FIXME !!!
   // This is not safe to assume it's 3 because we're passing in a pointer
   // and we do NOT know what the actuial size is. I'll have to improve it.
   //
   for ( size_t i=0; i<3; i++ )
   {
      typeSet[i].clear();
      for ( size_t j=0; j<typeIdx[i].size(); j++ )
      {

	 // HEPEUP and Py8 event record are shifted by 3 
	 // (system particle + 2 beam particles in Py8 event)
	 // ... EXCEPT FOR THE DECAY PRODUCTS IF DONE AT THE ME LEVEL !!!!!
	 // ... in such case, the decay productes get gutted out of the sequence 
	 // and get placed in Py8 Event later on...
	 // so we need to figure out the shift
         int pos = typeIdx[i][j]; 
	 int shift = 3;
	 // skip the first 2 entrirs in HEPEUP - they're incoming partons
	 for ( int ip=2; ip<pos; ip++ )
	 {
	    // alternative can be: moth1 != 1 && moth2 !=2...
	    // but moth1 == moth2 means pointer to the same mother t
	    // that can only be a resonance, unless moth1==moth2==0
	    //
	    if ( hepeup.MOTHUP[ip].first == hepeup.MOTHUP[ip].second )
	    {
	       shift--;
	    }	 
	 }	 
	 pos += shift;
	 // typeSet[i].insert( event[pos].daughter1() );
	 typeSet[i].insert( pos );
      }
   }   
  
  // int iTypeEnd = (typeIdx[1].empty()) ? 1 : 2;

// --> FIXME !!!
   int iType = 0; // only LIGHT jets for now
   int jetAllow = 0; // hardcoded for now for the same value as is set in Steve's example
   // at present, not even in use...
   // int jetMatch = 0; // hardcoded for now for the same value as is set in Steve's example
   
  // Loop over particles and decide what to pass to the jet algorithm
  for (int i = 0; i < workEventJet.size(); ++i) 
  {

    if (!workEventJet[i].isFinal()) continue;

    // jetAllow option to disallow certain particle types
    if (jetAllow == 1) 
    {

      // Original AG+Py6 algorithm explicitly excludes tops,
      // leptons and photons.
      int id = workEventJet[i].idAbs();
      if ((id >= 11 && id <= 16) || id == ID_TOP || id == ID_PHOTON) 
      {
        workEventJet[i].statusNeg();
        continue;
      }
    }

    // Get the index of this particle in original event
    int idx = workEventJet[i].daughter1();

    // Start with particle idx, and afterwards track mothers
    while (true) 
    {

      // Light jets
      if (iType == 0) 
      {

        // Do not include if originates from heavy jet or 'other'
        if (typeSet[1].find(idx) != typeSet[1].end() ||
            typeSet[2].find(idx) != typeSet[2].end()) 
        {
          workEventJet[i].statusNeg();
          break;
        }

        // Made it to start of event record so done
        if (idx == 0) 
	{
	   break;
	}
        // Otherwise next mother and continue
        idx = event[idx].mother1();

      // Heavy jets
      } 
      else if (iType == 1) 
      {

        // Only include if originates from heavy jet
        if (typeSet[1].find(idx) != typeSet[1].end()) break;

        // Made it to start of event record with no heavy jet mother,
        // so DO NOT include particle
        if (idx == 0) 
	{
          workEventJet[i].statusNeg();
          break;
        }

        // Otherwise next mother and continue
        idx = event[idx].mother1();

      } // if (iType)
    } // while (true)
  } // for (i)

  // For jetMatch = 2, insert ghost particles corresponding to
  // each hard parton in the original process
/*
  if (jetMatch > 0) 
  {

    for (int i = 0; i < int(typeIdx[iType].size()); i++) 
    {
      // Get y/phi of the parton
      Vec4   pIn = eventProcess[typeIdx[iType][i]].p();
      double y   = Vec4y(pIn);
      double phi = pIn.phi();

      // Create a ghost particle and add to the workEventJet
      double e   = MG5_GHOSTENERGY;
      double e2y = exp(2. * y);
      double pz  = e * (e2y - 1.) / (e2y + 1.);
      double pt  = sqrt(e*e - pz*pz);
      double px  = pt * cos(phi);
      double py  = pt * sin(phi);
      workEventJet.append(Particle(ID_GLUON, 99, 0, 0, 0, 0, 0, 0,
                                px, py, pz, e, 0., 0, 9.));

    } // for (i)
  } // if (jetMatch == 2)
*/

   for ( int i=0; i<workEventJet.size(); i++ )
   {
       
       // fisrt, weed out all entries marked with statusNeg();
       //
       if ( workEventJet[i].status() < 0 ) continue;
       
       
       // now weed out all entries above etaMax 
       // in principle, we can use etaJetMaxAlgo because it's set equal to etaJetMax
       // as for etaJetMax, it gets set to memain_.etaclmax
       //
       if ( fabs(workEventJet[i].eta()) > fJetEtaMax ) continue ; 
       
       // need to double check if native FastJet understands Py8 Event format
       // in general, PseudoJet gets formed from (px,py,pz,E)
       //
       fastjet::PseudoJet partTmp = workEventJet[i];
       fJetInput.push_back( partTmp );       
   }
      
   return fJetInput;

}


int Py8toJetInput::getAncestor( int pos, const Event& fullEvent, const Event& workEvent )
{

   int parentId = fullEvent[pos].mother1();
   int parentPrevId = 0;
   int counter = pos;
   
   while ( parentId > 0 )
   {               
         if ( parentId == fullEvent[counter].mother2() ) // carbon copy, keep walking up
	 {
	    parentPrevId = parentId;
	    counter = parentId;
	    parentId = fullEvent[parentPrevId].mother1();
	    continue;
	 }
	 
	 // we get here if not a carbon copy
	 
	 // let's check if it's a normal process, etc.
	 //
	 if ( (parentId < parentPrevId) || parentId < fullEvent[counter].mother2() ) // normal process
	 {
	    
	    // first of all, check if hard block
	    if ( abs(fullEvent[counter].status()) == 22 || abs(fullEvent[counter].status()) == 23 )
	    {
	       // yes, it's the hard block
	       // we got what we want, and can exit now !
	       parentId = counter;
	       break;
	    }
	    else
	    {
	       parentPrevId = parentId;
	       parentId = fullEvent[parentPrevId].mother1();
	    }
	 }
	 else if ( parentId > parentPrevId || parentId > pos ) // "circular"/"forward-pointing parent" - intermediate process
	 {
	    parentId = -1;
	    break;
	 }

         // additional checks... although we shouldn't be geeting here all that much...
	 //	 
	 if ( abs(fullEvent[parentId].status()) == 22 || abs(fullEvent[parentId].status())== 23 ) // hard block
	 {
	    break;
	 } 	 
	 if ( abs(fullEvent[parentId].status()) < 22 ) // incoming
	 {
	    parentId = -1;
	    break;
	 } 
   }
   
   return parentId;

}

#include "HepMC/HEPEVT_Wrapper.h"
#include <cassert>

const std::vector<fastjet::PseudoJet> 
Py8toJetInputHEPEVT::fillJetAlgoInput( const Event& event, const Event& workEvent, 
                                       const lhef::LHEEvent* lhee,
                                       const std::vector<int>* )
{

   fJetInput.clear();

   HepMC::HEPEVT_Wrapper::zero_everything();   
      
   // service container for further mother-daughters links
   //
   std::vector<int> Py8PartonIdx; // position of original (LHE) partons in Py8::Event
   Py8PartonIdx.clear(); 
   std::vector<int> HEPEVTPartonIdx; // position of LHE partons in HEPEVT (incl. ME-generated decays)
   HEPEVTPartonIdx.clear(); 

   // general counter
   //
   int index = 0;

   int Py8PartonCounter = 0;
   int HEPEVTPartonCounter = 0;
   
   // find the fisrt parton that comes from LHE (ME-generated)
   // skip the incoming particles/partons
   for ( int iprt=1; iprt<event.size(); iprt++ )
   {
      const Particle& part = event[iprt];
      if ( abs(part.status()) < 22 ) continue; // below 10 is "service"
                                               // 11-19 are beam particles; below 10 is "service"
					       // 21 is incoming partons      
      Py8PartonCounter = iprt;
      break;
   }

   const lhef::HEPEUP& hepeup = *lhee->getHEPEUP();
   // start the counter from 2, because we don't want the incoming particles/oartons !
   for ( int iprt=2; iprt<hepeup.NUP; iprt++ )
   {
      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, hepeup.IDUP[iprt] );
      HepMC::HEPEVT_Wrapper::set_status( index, 2 );
      HepMC::HEPEVT_Wrapper::set_momentum( index, hepeup.PUP[iprt][0], hepeup.PUP[iprt][1], hepeup.PUP[iprt][2], hepeup.PUP[iprt][4] );
      HepMC::HEPEVT_Wrapper::set_mass( index, hepeup.PUP[iprt][4] );
      // --> FIXME HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 ); // NO, not anymore to the "system particle"
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 ); 
      if (  hepeup.MOTHUP[iprt].first > 2 && hepeup.MOTHUP[iprt].second > 2 ) // decay from LHE, will NOT show at the start of Py8 event !!!
      {
         HEPEVTPartonCounter++;
	 continue;
      }
      Py8PartonIdx.push_back( Py8PartonCounter );
      Py8PartonCounter++;
      HEPEVTPartonIdx.push_back( HEPEVTPartonCounter);
      HEPEVTPartonCounter++;   
   }
      
   HepMC::HEPEVT_Wrapper::set_number_entries( index );   
         
   // now that the initial partons are in, attach parton-level from Pythia8
   // do NOT reset index as we need to *add* more particles sequentially
   //
   for ( int iprt=1; iprt<workEvent.size(); iprt++ ) // from 0-entry (system) or from 1st entry ???
   {
   
      const Particle& part = workEvent[iprt];
      

//      if ( part.status() != 62 ) continue;
      if ( part.status() < 51 ) continue;
      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, part.id() );
      
      // HepMC::HEPEVT_Wrapper::set_status( index, event.statusHepMC(iprt) ); 
      HepMC::HEPEVT_Wrapper::set_status( index, 1 );      
      HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), part.pz(), part.e() );
      HepMC::HEPEVT_Wrapper::set_mass( index, part.m() );
      HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 ); // just set to 0 like in Py6...
                                                         // although for some, mother will need to be re-set properly !
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 );

      // now refine mother-daughters links, where applicable
      
      int parentId = getAncestor( part.daughter1(), event, workEvent );
      
      if ( parentId <= 0 ) continue;

      for ( int idx=0; idx<(int)Py8PartonIdx.size(); idx++ )
      {
         if ( parentId == Py8PartonIdx[idx] )
	 {
            int idx1 = HEPEVTPartonIdx[idx]; 
	    HepMC::HEPEVT_Wrapper::set_parents( index, idx1+1, idx1+1 ); 
	    break;
	 }
      }

   } 
        
   HepMC::HEPEVT_Wrapper::set_number_entries( index );

// --> FIXME   HepMC::HEPEVT_Wrapper::set_event_number( fEventNumber ); // well, if you know it... well, it's one of the counters...
      
//   HepMC::HEPEVT_Wrapper::print_hepevt();
   
   return fJetInput;

}
