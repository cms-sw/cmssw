#include "OutputDDToDDL.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
// #include <CondCore/DBOutputService/interface/PoolDBOutputService.h>
#include <FWCore/Framework/interface/ESHandle.h>

//#include <DetectorDescription/Core/interface/DDMaterial.h>
//#include <DetectorDescription/Core/interface/DDTransform.h>
//#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDLogicalPart.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/Core/interface/DDRoot.h>
#include <DetectorDescription/Core/interface/DDName.h>
#include <DetectorDescription/Core/interface/DDPosData.h>
//#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <DetectorDescription/Core/interface/DDPartSelection.h>
// #include <DetectorDescription/PersistentDDDObjects/interface/DDDToPersFactory.h>
#include <DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h>
#include <Geometry/Records/interface/IdealGeometryRecord.h>

// for clhep stuff..
//#include <DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
//#include <set>
//#include <map>
#include <utility>

OutputDDToDDL::OutputDDToDDL(const edm::ParameterSet& iConfig) : fname_()
{
  //  std::cout<<"OutputDDToDDL::OutputDDToDDL"<<std::endl;
  rotNumSeed_ = iConfig.getParameter<int>("rotNumSeed");
  fname_ = iConfig.getUntrackedParameter<std::string>("fileName");
  if ( fname_ == "" ) {
    xos_ = &std::cout;
  } else {
    xos_ = new std::ofstream(fname_.c_str());
  }
  (*xos_) << "<?xml version=\"1.0\"?>" << std::endl;
  (*xos_) << "<DDDefinition xmlns=\"http://www.cern.ch/cms/DDL\"" << std::endl;
  (*xos_) << " xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"" << std::endl;
  (*xos_) << "xsi:schemaLocation=\"http://www.cern.ch/cms/DDL ../../../DetectorDescription/Schema/DDLSchema.xsd\">" << std::endl;
  (*xos_) << std::fixed << std::setprecision(18);
}
OutputDDToDDL::~OutputDDToDDL()
{
  (*xos_) << "</DDDefinition>" << std::endl;
  (*xos_) << std::endl;
  xos_->flush();

}

void
OutputDDToDDL::beginJob( edm::EventSetup const& es) 
{
  std::cout<<"OutputDDToDDL::beginJob"<<std::endl;

  edm::ESHandle<DDCompactView> pDD;

  es.get<IdealGeometryRecord>().get( pDD );

  DDCompactView::DDCompactView::graph_type gra = pDD->graph();
  // temporary stores:
  std::set<DDLogicalPart> lpStore;
  std::set<DDMaterial> matStore;
  std::set<DDSolid> solStore;
  //  std::vector<DDSpecifics> specStore; // are specs unique? I believe so, or else all sorts of bad things happen.
  //  std::vector<std::pair< DDPartSelection*, DDsvalues_type* > > specStore;
  std::map<DDsvalues_type, std::vector<DDPartSelection*>, ddsvaluesCmp > specStore;
  //  std::set<std::pair< DDPartSelection*, DDsvalues_type* > > specStore;
  std::set<DDRotation> rotStore;

  DDCoreToDDXMLOutput out;
  
  std::string rn = fname_;
  size_t foundLastDot= rn.find_last_of('.');
  size_t foundLastSlash= rn.find_last_of('/');
  if ( foundLastSlash > foundLastDot && foundLastSlash != std::string::npos) {
    std::cout << "What? last . before last / in path for filename... this should die..." << std::endl;
  }
  if ( foundLastDot != std::string::npos && foundLastSlash != std::string::npos ) {
    out.ns_ = rn.substr(foundLastSlash,foundLastDot);
  } else if ( foundLastDot != std::string::npos ) {
    out.ns_ = rn.substr(0, foundLastDot);
  } else {
    std::cout << "What? no file name? Attempt at namespace =\"" << out.ns_ << "\" filename was " << fname_ <<  std::endl;
  }
  std::cout << "fname_=" << fname_ << " namespace = " << out.ns_ << std::endl;
  std::string ns_ = out.ns_;

  (*xos_) << std::fixed << std::setprecision(18);
  typedef  DDCompactView::graph_type::const_adj_iterator adjl_iterator;

  adjl_iterator git = gra.begin();
  adjl_iterator gend = gra.end();    
    
  DDCompactView::graph_type::index_type i=0;
  (*xos_) << "<PosPartSection label=\"" << ns_ << "\">" << std::endl;
  git = gra.begin();
  for (; git != gend; ++git) 
    {
      const DDLogicalPart & ddLP = gra.nodeData(git);
      if ( lpStore.find(ddLP) != lpStore.end() ) {
// 	specStore.reserve(specStore.size()+ddLP.attachedSpecifics().size());
// 	//      std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::iterator spit(ddLP.attachedSpecifics().begin()), spend (ddLP.attachedSpecifics().end());
// 	//       while ( spit != spend ) {
// 	// 	specStore.push_back(std::make_pair(spit->first, spit->second));
// 	// 	++spit;
// 	//       }
// 	std::copy(ddLP.attachedSpecifics().begin(), ddLP.attachedSpecifics().end(), std::back_inserter(specStore));//, specStore.end()); //
// attached specifics are std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >
	std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::const_iterator spit(ddLP.attachedSpecifics().begin()), spend(ddLP.attachedSpecifics().end());
	for ( ; spit != spend; ++spit ) {
	  const DDsvalues_type & ds = *spit->second;
	  std::map<DDsvalues_type, std::vector<DDPartSelection*> >::const_iterator fit = specStore.find(ds);
	  if ( fit == specStore.end() ) {
	    std::vector<DDPartSelection*> psv;
	    psv.push_back(spit->first);
	    specStore[ds] = psv;
	  } else {
	    std::vector<DDPartSelection*> psv(specStore[ds]);
	    psv.push_back(spit->first);
	    specStore[ds] = psv;
	  }
	}
      }
      lpStore.insert(ddLP);
      addToMatStore( ddLP.material(), matStore );
      addToSolStore( ddLP.solid(), solStore, rotStore );
      //      specStore.insert(ddLP.specifics());
      //    DDSpecifics specs(ddLP.specifics());
      //      specStore.push_back(specs);
      ++i;
      if (git->size()) 
	{
	  // ask for children of ddLP  
	  DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
	  DDCompactView::graph_type::edge_list::const_iterator cend = git->end();
	  for (; cit != cend; ++cit) 
	    {
	      const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
	      if (lpStore.find(ddcurLP) != lpStore.end()) {
// 		specStore.reserve(specStore.size()+ddcurLP.attachedSpecifics().size());
// 		std::copy(ddcurLP.attachedSpecifics().begin(), ddcurLP.attachedSpecifics().end(), std::back_inserter(specStore));
		std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::const_iterator spit(ddcurLP.attachedSpecifics().begin()), spend(ddcurLP.attachedSpecifics().end());
		for ( ; spit != spend; ++spit ) {
		  const DDsvalues_type & ds = *spit->second;
		  std::map<DDsvalues_type, std::vector<DDPartSelection*> >::const_iterator fit = specStore.find(ds);
		  if ( fit == specStore.end() ) {
		    std::vector<DDPartSelection*> psv;
		    psv.push_back(spit->first);
		    specStore[ds] = psv;
		  } else {
		    std::vector<DDPartSelection*> psv(specStore[ds]);
		    psv.push_back(spit->first);
		  }
		}
	      }
	      lpStore.insert(ddcurLP);
	      addToMatStore(ddcurLP.material(), matStore);
	      addToSolStore(ddcurLP.solid(), solStore, rotStore);
	      rotStore.insert(gra.edgeData(cit->second)->rot_);
	      //	      specStore.insert(ddcurLP.specifics());
	      //	      specStore.push_back(ddcurLP.specifics());
	      out.position(ddLP, ddcurLP, gra.edgeData(cit->second), rotNumSeed_, *xos_);
	    } // iterate over children
	} // if (children)
    } // iterate over graph nodes  
  (*xos_) << "</PosPartSection>" << std::endl;
  
  (*xos_) << std::scientific << std::setprecision(18);
  std::set<DDMaterial>::const_iterator it(matStore.begin()), ed(matStore.end());
  (*xos_) << "<MaterialSection label=\"" << ns_ << "\">" << std::endl;
  for (; it != ed; ++it) {
    if (! it->isDefined().second) continue;
    out.material(*it, *xos_);
  }
  (*xos_) << "</MaterialSection>" << std::endl;

  (*xos_) << "<RotationSection label=\"" << ns_ << "\">" << std::endl;
  (*xos_) << std::fixed << std::setprecision(18);
  std::set<DDRotation>::iterator rit(rotStore.begin()), red(rotStore.end());
  for (; rit != red; ++rit) {
    if (! rit->isDefined().second) continue;
    if ( rit->toString() != ":" ) {
      DDRotation r(*rit);
      out.rotation(r, *xos_);
    }
  } 
  (*xos_) << "</RotationSection>" << std::endl;

  (*xos_) << std::fixed << std::setprecision(18);
  std::set<DDSolid>::const_iterator sit(solStore.begin()), sed(solStore.end());
  (*xos_) << "<SolidSection label=\"" << ns_ << "\">" << std::endl;
  for (; sit != sed; ++sit) {
    if (! sit->isDefined().second) continue;  
    out.solid(*sit, *xos_);
  }
  (*xos_) << "</SolidSection>" << std::endl;

  std::set<DDLogicalPart>::iterator lpit(lpStore.begin()), lped(lpStore.end());
  (*xos_) << "<LogicalPartSection label=\"" << ns_ << "\">" << std::endl;
  for (; lpit != lped; ++lpit) {
    if (! lpit->isDefined().first) continue;  
    const DDLogicalPart & lp = *lpit;
      out.logicalPart(lp, *xos_);
  }
  (*xos_) << "</LogicalPartSection>" << std::endl;

  (*xos_) << std::fixed << std::setprecision(18);
//   std::vector<std::string> partSelections;
//   std::map<std::string, std::vector<std::pair<std::string, double> > > values;
//   std::map<std::string, int> isEvaluated;
  
  //  DDSpecifics::iterator<DDSpecifics> spit(DDSpecifics::begin()), spend(DDSpecifics::end());
  //  std::set<DDSpecifics>::iterator spit(specStore.begin()), spend (specStore.end());
  //std::vector<DDSpecifics>::iterator spit(specStore.begin()), spend (specStore.end());

  // prior to output we should somehow clean up the store and use a new store.
  // the new store should be something like many part-selections to many svalues...
  // if you have the same svalues and different partselections, store these together.

  // first, remove duplicates.
  /// LASTMESSWORKING
//   std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::iterator spit(specStore.begin()), spend (specStore.end());
//   std::vector<std::pair<DDPartSelection*, DDsvalues_type*> > newSpecStore;
//   std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::iterator spit2, spend2;
//   //  std::vector<std::pair<std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::const_iterator, DDPartSelection*> > specMatches; 
//   //  std::vector<std::pair<DDsvalues_type*, std::vector<DDPartSelection*> > > specMatches; 
//   std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > > specMatches; 
//   //  std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > > spm2;
//   bool alwaysmatchedboth(true);
//   bool partselsmatch(false);
//   bool valuesmatch(false);
//   bool foundamatch(false);
//   //  size_t ind = 0;
//   std::cout << "size of store is " << specStore.size() << std::endl;
//   while ( spit != spend ) {
//     if (newSpecStore.size() == 0) {
//       std::cout << "adding first one!" << std::endl;
//       newSpecStore.push_back(*spit);
//     } else {
//       spit2 = newSpecStore.begin();
//       spend2 = newSpecStore.end();
//       foundamatch = false;
//       while ( spit2 != spend2 ) {
// 	partselsmatch = doPartSelectionsMatch (spit->first, spit2->first);
// 	valuesmatch = doDDsvaluesMatch( spit->second, spit2->second);
// 	if ( spit->first->begin()->lp_.toString() == "muonBase:MUON" ) {
// 	  std::cout << *(spit->first) << " and " << (valuesmatch ? " values match " : " values don't match ")
// 		    << (partselsmatch ? " part selections match " : " part selections don't match ") << std::endl;
// 	  std::cout <<" just what is a DDsvalues_type?" << std::endl;
// 	  //	std::vector< std::pair<unsigned int, DDValue> > DDsvalues_type;   
// 	  //	std::vector<std::pair<unsigned int, DDValue> >::const_iterator dsit2(nssit->first->begin()), dsendit2(nssit->first->end());
// 	  DDsvalues_type::const_iterator dsit2(spit->second->begin()), dsendit2(spit->second->end());
// 	  for ( ; dsit2 != dsendit2; ++dsit2 ) {
// 	    std::cout << "  " << dsit2->first << " is " << dsit2->second.name() << " and has " << dsit2->second.strings().size() << " string values and "
// 		      << (dsit2->second.isEvaluated() ? "is evaluated. " : "is NOT evaluated.") << std::endl;
// 	    for (size_t i = 0; i < dsit2->second.strings().size(); ++i) {
// 	      if (dsit2->second.isEvaluated()) {
// 		std::cout << dsit2->second.doubles()[i] << " ";
// 	      } else {
// 		std::cout << dsit2->second.strings()[i] << " ";
// 	      }
// 	    }
// 	    std::cout << std::endl;
// 	  }
// 	}
// 	if  (valuesmatch && !partselsmatch) {
// 	  alwaysmatchedboth = false;
// 	  std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > >::iterator bsmit(specMatches.begin()), bsmendit(specMatches.end());
// 	  while ( bsmit != bsmendit ) {
// 	    if ( bsmit->first.find(spit->second) != bsmit->first.end() ) {
// 	      bsmit->second.insert(spit->first);
// 	      break;
// 	    }
// 	    ++bsmit;
// 	  }
// 	  if ( bsmit == bsmendit || specMatches.size() == 0 ) {
// 	    std::set<DDPartSelection*> psvect;
// 	    psvect.insert(spit->first);
// 	    std::set<DDsvalues_type*> sval;
// 	    sval.insert(spit->second);
// 	    specMatches.push_back( std::make_pair(sval, psvect) );
// 	  }
// 	} else if (!valuesmatch && partselsmatch) {
// 	  std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > >::iterator bsmit(specMatches.begin()), bsmendit(specMatches.end());
// 	  while ( bsmit != bsmendit ) {

// 	    if ( bsmit->second.find(spit->first) != bsmit->second.end() ) {
// 	      bsmit->first.insert(spit->second);
// 	      break;
// 	    }
// 	    ++bsmit;
// 	  }
// 	  if ( bsmit == bsmendit || specMatches.size() == 0 ) {
// 	    std::set<DDPartSelection*> psvect;
// 	    psvect.insert(spit->first);
// 	    std::set<DDsvalues_type*> sval;
// 	    sval.insert(spit->second);
// 	    specMatches.push_back( std::make_pair(sval, psvect) );
// 	  }
	  
// 	  // 	std::set<DDPartSelection*> psvect;
// 	  // 	psvect.insert(spit->first);
// 	  // 	specMatches.push_back( std::make_pair(spit->second, psvect) );
// 	  alwaysmatchedboth = false;
// 	}
// 	if (valuesmatch && partselsmatch) {
// 	  foundamatch = true;
// 	  break;
// 	}
// 	++spit2;
//       }
//       if (!foundamatch)
// 	newSpecStore.push_back(*(spit));
//     }
//     ++spit;
//   }
//   std::cout << "unduplicated = " << newSpecStore.size() << std::endl;
//   std::cout << "matches = " << specMatches.size() << std::endl;
//   if (alwaysmatchedboth) std::cout << "always matched both" << std::endl; else std::cout << "did NOT always match both" << std::endl;
/// LASTMESSWORKINGEND

//   std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::iterator nssit(newSpecStore.begin()), nssend (newSpecStore.end());
//   std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > > spm2;
//   for (; nssit != nssend; ++nssit) {
//     if  ( spm2.size() == 0 ) {
//       std::set<DDsvalues_type*> ssvt;
//       std::set<DDPartSelection*> sps;
//       ssvt.insert(nssit->second);
//       sps.insert(nssit->first);
//       spm2.push_back(std::make_pair(ssvt, sps));
//       continue;
//     } 
//     //else {
//     std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > >::iterator mit(spm2.begin()), mend (spm2.end()); 
//     for (; mit != mend; ++mit) {
//       if (mit->second.size() == 1 && doPartSelectionsMatch(*(mit->second.begin()), nssit->first) ) {
// 	mit->first.insert(nssit->second);
// 	std::cout <<" just what is a DDsvalues_type?" << std::endl;
// 	//	std::vector< std::pair<unsigned int, DDValue> > DDsvalues_type;   
// 	//	std::vector<std::pair<unsigned int, DDValue> >::const_iterator dsit2(nssit->first->begin()), dsendit2(nssit->first->end());
// 	DDsvalues_type::const_iterator dsit2(nssit->second->begin()), dsendit2(nssit->second->end());
// 	for ( ; dsit2 != dsendit2; ++dsit2 ) {
// 	  std::cout << "  " << dsit2->first << " is " << dsit2->second.name() << " and has " << dsit2->second.strings().size() << " string values and "
// 		    << (dsit2->second.isEvaluated() ? "is evaluated. " : "is NOT evaluated.") << std::endl;
// 	  for (size_t i = 0; i < dsit2->second.strings().size(); ++i) {
// 	    if (dsit2->second.isEvaluated()) {
// 	      std::cout << dsit2->second.doubles()[i] << " ";
// 	    } else {
// 	      std::cout << dsit2->second.strings()[i] << " ";
// 	    }
// 	  }
// 	  std::cout << std::endl;
// 	}
// 	break;
//       }
//     }
// //       std::set<DDPartSelection*>::iterator bpsit(mit->second.begin()), bpsitend(mit->second.end());
// //       for ( ; bpsit != bpsitend ; ++bpsit ) {
// // 	if ( !doPartSelectionsMatch (nssit->first, bpsit) ) { //mit->second.find(nssit->first) != mit->second.end() ) {
// // 	  break;
// // 	}	  
// //       }
// //       if (!doPartSelectionsMatch (nssit->first, bpsit)
// //     }

 
// //       if ( mit->first.find(nssit->second) != mit->first.end() ) {
// // 	mit->second.insert(nssit->first);
// //       }
// //    }
//     if ( mit == mend ) {
//       std::set<DDsvalues_type*> ssvt;
//       std::set<DDPartSelection*> sps;
//       ssvt.insert(nssit->second);
//       sps.insert(nssit->first);
//       spm2.push_back(std::make_pair(ssvt, sps));
//       continue;
//     }
//   }

//  std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > >::const_iterator mit(specMatches.begin()), mend (specMatches.end()); 
  // std::vector<std::pair<std::set<DDsvalues_type*>, std::set<DDPartSelection*> > >::const_iterator mit(spm2.begin()), mend (spm2.end()); 
  //    std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::const_iterator mit(newSpecStore.begin()), mend (newSpecStore.end());
  std::map<DDsvalues_type, std::vector<DDPartSelection*> >::const_iterator mit(specStore.begin()), mend (specStore.end());
  (*xos_) << "<SpecParSection label=\"" << ns_ << "\">" << std::endl;
  for (; mit != mend; ++mit) {
    //    if ( !spit->isDefined().second ) continue;
    out.specpar ( *mit, *xos_ );
  } 
  (*xos_) << "</SpecParSection>" << std::endl;

}

bool OutputDDToDDL::doPartSelectionsMatch( DDPartSelection* sp1, DDPartSelection* sp2 ) {
//       const std::vector<DDPartSelectionLevel>& psl1 (*(spit->first));
//       const std::vector<DDPartSelectionLevel>& psl2 (*(spit2->first));
  const std::vector<DDPartSelectionLevel>& psl1 (*sp1);
  const std::vector<DDPartSelectionLevel>& psl2 (*sp2);
  bool retval = true;
  if (psl1.size() != psl2.size()) retval = false;
  size_t ind = 0;
  while ( retval && ind < psl1.size() ) {
    if ( psl1[ind].lp_.toString() != psl2[ind].lp_.toString() ) retval = false;
    if ( psl1[ind].copyno_ != psl2[ind].copyno_ ) retval = false;
    if ( psl1[ind].selectionType_ != psl2[ind].selectionType_ ) retval = false;
    ++ind;
  }
  return retval;
}

bool OutputDDToDDL::doDDsvaluesMatch ( DDsvalues_type* sv1, DDsvalues_type* sv2 ) {
  bool valuesmatch = true;
  if ( sv1->size() != sv2->size() ) valuesmatch=false;
  size_t ind = 0;
  while ( valuesmatch && ind < sv1->size() ) {
    if ( (*sv1)[ind].first != (*sv2)[ind].first ) valuesmatch=false;
    if ( (*sv1)[ind].second != (*sv2)[ind].second ) valuesmatch=false;
    ++ind;
  }
  return valuesmatch;
}

void  OutputDDToDDL::addToMatStore( const DDMaterial& mat, std::set<DDMaterial> & matStore) {
  matStore.insert(mat);
  if ( mat.noOfConstituents() != 0 ) {
    DDMaterial::FractionV::value_type frac;
    int findex(0);
    while ( findex < mat.noOfConstituents() ) {
      if ( matStore.find(mat.constituent(findex).first) == matStore.end() ) {
	addToMatStore( mat.constituent(findex).first, matStore );
      }
      ++findex;
    }
  }
}

void  OutputDDToDDL::addToSolStore( const DDSolid& sol, std::set<DDSolid> & solStore, std::set<DDRotation>& rotStore ) {
      solStore.insert(sol);
      if ( sol.shape() == ddunion || sol.shape() == ddsubtraction || sol.shape() == ddintersection ) {
	const DDBooleanSolid& bs (sol);
	if ( solStore.find(bs.solidA()) == solStore.end()) {
	  addToSolStore(bs.solidA(), solStore, rotStore);
	}
	if ( solStore.find(bs.solidB()) == solStore.end()) {
	  addToSolStore(bs.solidB(), solStore, rotStore);
	}
	rotStore.insert(bs.rotation());
      }
}

bool ddsvaluesCmp::operator() ( const  DDsvalues_type& sv1, const DDsvalues_type& sv2 ) {
  if ( sv1.size() < sv2.size() ) return true;
  size_t ind = 0;
  for (; ind < sv1.size(); ++ind) {
    if ( sv1[ind].first >= sv2[ind].first ) return false;
    if ( !(sv1[ind].second < sv2[ind].second) ) return false;
  }
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OutputDDToDDL);
