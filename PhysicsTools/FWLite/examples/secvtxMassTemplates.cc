// -*- C++ -*-

// CMS includes
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/FlavorHistory.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Math/GenVector/PxPyPzM4D.h"

#include "PhysicsTools/FWLite/interface/EventContainer.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h" 

// Root includes
#include "TROOT.h"
#include "TH2F.h"

using namespace std;
using optutl::CommandLineParser;

enum
{
   kNormalMode,
   kVqqMode,
   kLFMode,
   kWcMode
};

//////////////////////////
// Forward Declarations //
//////////////////////////

// This subroutine, written by you (below), uses the command line
// arguments and creates an output tag (if any).  This subroutine must
// exist.
void outputNameTagFunc (string &tag);

// Book all histograms to be filled this job.  If wanted, you can skip
// this subroutine and book all histograms in the main subroutine.
void bookHistograms (fwlite::EventContainer &eventCont);

// Calculate the name that should be used for this event based on the
// mode, the HF word, and (if necessary), whether or not it's a W or
// Z.  Returns false if the event should not be processed.
bool calcSampleName (fwlite::EventContainer &eventCont, string &sampleName);

///////////////////////////
// ///////////////////// //
// // Main Subroutine // //
// ///////////////////// //
///////////////////////////

int main (int argc, char* argv[]) 
{
   ////////////////////////////////
   // ////////////////////////// //
   // // Command Line Options // //
   // ////////////////////////// //
   ////////////////////////////////

   // Tell people what this analysis code does and setup default options.
   CommandLineParser parser ("Creates SecVtx Mass templates");

   //////////////////////////////////////////////////////
   // Add any command line options you would like here //
   //////////////////////////////////////////////////////
   parser.addOption ("mode",         CommandLineParser::kInteger, 
                     "Normal(0), VQQ(1), LF(2), Wc(3)", 
                     0);   
   parser.addOption ("sampleName",   CommandLineParser::kString, 
                     "Sample name (e.g., top, Wqq, etc.)");   

   ////////////////////////////////////////////////
   parser.stringValue ("outputFile") = "jetPt"; // .root added automatically

   // Parse the command line arguments
   parser.parseArguments (argc, argv);

   //////////////////////////////////
   // //////////////////////////// //
   // // Create Event Container // //
   // //////////////////////////// //
   //////////////////////////////////

   // This object 'eventCont' is used both to get all information from the
   // eventCont as well as to store histograms, etc.
   // Parse the command line arguments
   parser.parseArguments (argc, argv);

   //////////////////////////////////
   // //////////////////////////// //
   // // Create Event Container // //
   // //////////////////////////// //
   //////////////////////////////////

   fwlite::EventContainer eventCont (parser);

   ////////////////////////////////////////
   // ////////////////////////////////// //
   // //         Begin Run            // //
   // // (e.g., book histograms, etc) // //
   // ////////////////////////////////// //
   ////////////////////////////////////////

   // Setup a style
   gROOT->SetStyle ("Plain");

   // Book those histograms!
   bookHistograms (eventCont);

   //////////////////////
   // //////////////// //
   // // Event Loop // //
   // //////////////// //
   //////////////////////

   for (eventCont.toBegin(); !eventCont.atEnd(); ++eventCont) 
   {
      //////////////////////////////////
      // Take What We Need From Event //
      //////////////////////////////////
      fwlite::Handle< vector< pat::Jet > > jetCollection;
      jetCollection.getByLabel (eventCont, "selectedLayer1Jets");
      assert ( jetCollection.isValid() );
						
      fwlite::Handle< vector< pat::Muon > > goodMuonCollection;
      goodMuonCollection.getByLabel (eventCont, "goodLeptons");
      assert ( goodMuonCollection.isValid() );
			
      // If we don't have any good leptons, don't bother
      if (! goodMuonCollection->size() )
      {
         continue;
      }

      // get the sample name for this event
      string sampleName;
      if ( ! calcSampleName (eventCont, sampleName) )
      {
         // We don't want this one.
         continue;
      }

      //////////////////////////////////////
      // Tagged Jets and Flavor Separator //
      //////////////////////////////////////
      int numBottom = 0, numCharm = 0, numLight = 0;
      int numTags = 0;
      double sumVertexMass = 0.;
      // Loop over the jets and find out which are tagged
      const vector< pat::Jet >::const_iterator kJetEnd = jetCollection->end();
      for (vector< pat::Jet >::const_iterator jetIter = jetCollection->begin(); 
           kJetEnd != jetIter; 
           ++jetIter) 
      {
         // Is this jet tagged and does it have a good secondary vertex
         if( jetIter->bDiscriminator("simpleSecondaryVertexBJetTags") < 2.05 )
         {
            // This jet isn't tagged
            continue;
         }
         reco::SecondaryVertexTagInfo const * svTagInfos 
            = jetIter->tagInfoSecondaryVertex("secondaryVertex");
         if ( svTagInfos->nVertices() <= 0 ) 
         {
            // Given that we are using simple secondary vertex
            // tagging, I don't think this should ever happen.
            // Maybe we should put a counter here just to check.
            continue;
         } // if we have no secondary verticies
         
         // count it
         ++numTags;

         // What is the flavor of this jet
         int jetFlavor = std::abs( jetIter->partonFlavour() );
         if (5 == jetFlavor)
         {
            ++numBottom;
         } // if bottom 
         else if (4 == jetFlavor)
         {
            ++numCharm;
         } // if charm
         else
         {
            ++numLight;
         } // if light flavor

         ///////////////////////////
         // Calculate SecVtx Mass //
         ///////////////////////////
         ROOT::Math::LorentzVector< ROOT::Math::PxPyPzM4D<double> > sumVec;
         reco::CompositeCandidate vertexCand;
         reco::Vertex::trackRef_iterator 
            kEndTracks = svTagInfos->secondaryVertex(0).tracks_end();
         for (reco::Vertex::trackRef_iterator trackIter = 
                 svTagInfos->secondaryVertex(0).tracks_begin(); 
              trackIter != kEndTracks; 
              ++trackIter ) 
         {
            const double kPionMass = 0.13957018;
            ROOT::Math::LorentzVector< ROOT::Math::PxPyPzM4D<double> > vec;
            vec.SetPx( (*trackIter)->px() );
            vec.SetPy( (*trackIter)->py() );
            vec.SetPz( (*trackIter)->pz() );
            vec.SetM (kPionMass);
            sumVec += vec;
         } // for trackIter
         sumVertexMass += sumVec.M();
         if (2 == numTags)
         {
            // We've got enough.  Stop.
            break;
         } // if we have enough tags
      } // for jet

      ////////////////////////
      // General Accounting //
      ////////////////////////
      int numJets = std::min( (int) jetCollection->size(), 5 );
      eventCont.hist( sampleName + "_jettag")->Fill (numJets, numTags);

      // If we don't have any tags, don't bother going on
      if ( ! numTags)
      {
         continue;
      }

      ///////////////////////////////////////
      // Calculate average SecVtx mass and //
      // fill appropriate histograms.      //
      ///////////////////////////////////////
      sumVertexMass /= numTags;
      string whichtag = "";
      if (1 == numTags)
      {
         // single tag
         if      (numBottom)              whichtag = "_b";
         else if (numCharm)               whichtag = "_c";
         else if (numLight)               whichtag = "_q";
         else                             whichtag = "_X";
      } else {
         // double tags
         if      (2 == numBottom)         whichtag = "_bb";
         else if (2 == numCharm)          whichtag = "_cc";
         else if (2 == numLight)          whichtag = "_qq";
         else if (numBottom && numCharm)  whichtag = "_bc";
         else if (numBottom && numLight)  whichtag = "_bq";
         else if (numCharm  && numLight)  whichtag = "_bq";
         else                             whichtag = "_XX";
      } // if two tags
      string massName = sampleName 
         + Form("_secvtxMass_%dj_%dt", numJets, numTags);
      eventCont.hist(massName)->Fill (sumVertexMass);
      eventCont.hist(massName + whichtag)->Fill (sumVertexMass);
   } // for eventCont
      
   ////////////////////////
   // ////////////////// //
   // // Clean Up Job // //
   // ////////////////// //
   ////////////////////////

   // Histograms will be automatically written to the root file
   // specificed by command line options.

   // All done!  Bye bye.
   return 0;
}


//////////////  //////////////////////////////////  //////////////
//////////////  // //////////////////////////// //  //////////////
//////////////  // // Supporting Subroutines // //  //////////////
//////////////  // //////////////////////////// //  //////////////
//////////////  //////////////////////////////////  //////////////


void outputNameTagFunc (string &tag)
{
   // If you do not want to give you output filename any "tag" based
   // on the command line options, simply do nothing here.  This
   // function is designed to be called by fwlite::EventContainer constructor.

   // if ( boolValue ("someCondition") )
   // { 
   //    tag += "_someCond";
   // }
}


void bookHistograms (fwlite::EventContainer &eventCont)
{
   /////////////////////////////////////////////
   // First, come up with all possible base   //
   // names (E.g., Wbb, Wb2, etc.).           //
   /////////////////////////////////////////////
   CommandLineParser &parser = eventCont.parser();
   CommandLineParser::SVec baseNameVec;
   CommandLineParser::SVec beginningVec, endingVec;
   switch ( parser.integerValue ("mode") )
   {
      case kVqqMode:
         // We want Wbb, Wb2, .., Zbb, ..  In this case, we completely
         // ignore the sampleName that was passed in.
         // Starts with
         beginningVec.push_back ("X");
         beginningVec.push_back ("W");
         beginningVec.push_back ("Z");
         // Ends with
         endingVec.push_back( "bb" );
         endingVec.push_back( "b2" );
         endingVec.push_back( "cc" );
         endingVec.push_back( "c2" );
         for (CommandLineParser::SVecConstIter outerIter = beginningVec.begin();
              beginningVec.end() != outerIter;
              ++outerIter)
         {
            for (CommandLineParser::SVecConstIter innerIter = endingVec.begin();
                 endingVec.end() != innerIter;
                 ++innerIter)
            {
               baseNameVec.push_back( *outerIter + *innerIter);
            } // for innerIter
         } // for outerIter
         break;
      case kLFMode:
         // just like the default case, except that we do have some
         // heavy flavor pieces here, too.
         baseNameVec.push_back(parser.stringValue ("sampleName") + "b3");
         baseNameVec.push_back(parser.stringValue ("sampleName") + "c3");
         // no break because to add just the name as well
      default:
         // We just want to use the sample name as it was given to us.
         baseNameVec.push_back(parser.stringValue ("sampleName"));
   } // for switch

   ////////////////////////////////////////
   // Now the different tagging endings. //
   ////////////////////////////////////////
   CommandLineParser::SVec singleTagEndingVec, doubleTagEndingVec;
   singleTagEndingVec.push_back ("_b");
   singleTagEndingVec.push_back ("_c");
   singleTagEndingVec.push_back ("_q");
   doubleTagEndingVec.push_back ("_bb");
   doubleTagEndingVec.push_back ("_cc");
   doubleTagEndingVec.push_back ("_qq");
   doubleTagEndingVec.push_back ("_bc");
   doubleTagEndingVec.push_back ("_bq");
   doubleTagEndingVec.push_back ("_cq");

   /////////////////////////////////////////
   // Finally, let's put it all together. //
   /////////////////////////////////////////
   for (CommandLineParser::SVecConstIter baseIter = baseNameVec.begin();
        baseNameVec.end() != baseIter;
        ++baseIter)
   {
      //////////////////////////////////////////////////////
      // For each flavor, one jet/tag counting histogram. //
      //////////////////////////////////////////////////////
      TString histName = *baseIter + "_jettag";
      eventCont.add( new TH2F( histName, histName, 
                               6, -0.5, 5.5,
                               3, -0.5, 2.5) );
      for (int jet = 1; jet <= 5; ++jet)
      {
         for (int tag = 1; tag <= 2; ++tag)
         {
            ////////////////////////////////////////////
            // For each jet/tag, a single secvtx mass //
            ////////////////////////////////////////////
            if (tag > jet) continue;
            histName = *baseIter + Form ("_secvtxMass_%dj_%dt", jet, tag);
            eventCont.add( new TH1F( histName, histName, 40, 0, 10) );
            CommandLineParser::SVec *vecPtr = &singleTagEndingVec;
            if (2 == tag)
            {
               vecPtr = &doubleTagEndingVec;
            } // double tag
            for (CommandLineParser::SVecConstIter tagIter = vecPtr->begin();
                 vecPtr->end() != tagIter;
                 ++tagIter)
            {
               ////////////////////////////////////////////////////
               // And different secvtx mass for each tag ending. //
               ////////////////////////////////////////////////////
               histName = *baseIter + Form ("_secvtxMass_%dj_%dt", jet, tag)
                  + *tagIter;
               eventCont.add( new TH1F( histName, histName, 40, 0, 10) );
            } // for tagIter
         } // for tag
      } // for jet
   } // for baseIter
}
					

bool calcSampleName (fwlite::EventContainer &eventCont, string &sampleName)
{
   // calculate sample name
   CommandLineParser &parser = eventCont.parser();
   sampleName = parser.stringValue  ("sampleName");
   int mode   = parser.integerValue ("mode");

   /////////////////
   // Normal Mode //
   //// /////////////
   if (kNormalMode == mode)
   {
      // all we want is the sample name, so in this case we're done.
      return true;
   }
   // Get the heavy flavor category
   fwlite::Handle< unsigned int > heavyFlavorCategory;
   heavyFlavorCategory.getByLabel (eventCont, "flavorHistoryFilter");
   assert ( heavyFlavorCategory.isValid() );
   int HFcat = (*heavyFlavorCategory);

   ///////////////////////
   // Light Flavor Mode //
   ///////////////////////
   if (kLFMode == mode)
   {
      // Wqq
      if (5 == HFcat)
      {
         sampleName += "b3";
      } else if (6 == HFcat)
      {
         sampleName += "c3";
      } else if (11 != HFcat)
      {
         // skip this event
         return false;
      } // else if ! 11
      return true;
   }

   /////////////
   // Wc Mode //
   /////////////
   if (kWcMode == mode)
   {
      // Wc
      if (4 != HFcat)
      {
         // skip this event
         return false;
      } // if not Wc
      return true;
   } // else if Wc

   //////////////
   // Vqq Mode //
   //////////////
   // MadGraph (at least as CMS has implemented it) has this _lovely_
   // feature that if the W or Z is far enough off-shell, it erases
   // the W or Z from the event record.  This means that in some
   // number of cases, we won't be able to tell whether this is a W or
   // Z event by looking for a W or Z in the GenParticle collection.
   // (We'll eventually have to be more clever).
   sampleName = "X";
   fwlite::Handle< vector< reco::GenParticle > > genParticleCollection;
   genParticleCollection.getByLabel (eventCont, "genParticles");
   assert ( genParticleCollection.isValid() );
   // We don't know if it is a W, a Z, or neither
   // Iterate over genParticles
   const vector< reco::GenParticle>::const_iterator 
      kGenPartEnd = genParticleCollection->end();
   for (vector< reco::GenParticle>::const_iterator gpIter =
           genParticleCollection->begin(); 
        gpIter != kGenPartEnd; ++gpIter ) 
   {
      if (gpIter->status() == 3 && std::abs(gpIter->pdgId()) == 23)
      {
         sampleName = "Z";
         break;
      }
      if (gpIter->status() == 3 && std::abs(gpIter->pdgId()) == 24)
      {
         sampleName = "W";
         break;
      }
   } // for  gpIter
   switch (HFcat)
   {
      // from:
      // https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideFlavorHistory
      //  1. W+bb with >= 2 jets from the ME (dr > 0.5)
      //  2. W+b or W+bb with 1 jet from the ME
      //  3. W+cc from the ME (dr > 0.5)
      //  4. W+c or W+cc with 1 jet from the ME
      //  5. W+bb with 1 jet from the parton shower (dr == 0.0)
      //  6. W+cc with 1 jet from the parton shower (dr == 0.0)
      //  7. W+bb with >= 2 partons but 1 jet from the ME (dr == 0.0)
      //  8. W+cc with >= 2 partons but 1 jet from the ME (dr == 0.0)
      //  9. W+bb with >= 2 partons but 2 jets from the PS (dr > 0.5)
      // 10. W+cc with >= 2 partons but 2 jets from the PS (dr > 0.5)
      // 11. Veto of all the previous (W+ light jets)
      case 1:
         sampleName += "bb";
         break;
      case 2:
         // Sometimes this is referred to as 'b' (e.g., 'Wb'), but I
         // am using the suffix '2' to keep this case clear for when
         // we have charm (see below).
         sampleName += "b2";
         break; 
      case 3:
         sampleName += "cc";
         break;
      case 4:
         // We want to keep this case clear from real W + single charm
         // produced (as opposed to two charm quarks produced and one
         // goes down the beampipe), so we use 'c2' instead of 'c'.
         sampleName += "c2";
         break;
      default:
         // we don't want the rest of the cases.  Return an empty
         // string so we know.
         return false;
   } // switch HFcat
   return true;
}
