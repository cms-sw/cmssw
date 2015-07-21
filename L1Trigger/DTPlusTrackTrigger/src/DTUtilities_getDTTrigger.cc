/*! \class DTUtilities
 *  \author Ignazio Lazzizzera
 *  \author Sara Vanini
 *  \author Nicola Pozzobon
 *  \brief Utilities of L1 DT + Track Trigger for the HL-LHC
 *  \date 2008, Dec 25
 */

#include "L1Trigger/DTPlusTrackTrigger/interface/DTUtilities.h"

/// Method to get the DTTrigger
void DTUtilities::getDTTrigger()
{
  /// L1 DT Trigger flags
  bool doneBTI     = false;
  bool doneTSTheta = false;

  /// BTI
  std::vector< DTBtiTrigData > theBTITrigs = theDTTrigger->BtiTrigs();
  std::vector< DTBtiTrigData >::const_iterator iterBTITrig;

  /// Loop over the BTI triggers
  DTBtiTrigger* aDTBti;
  for ( iterBTITrig = theBTITrigs.begin();
        iterBTITrig != theBTITrigs.end();
        iterBTITrig++ )
  {
    /// Get the position and direction of the BTI
    Global3DPoint pos = theDTTrigger->CMSPosition( &(*iterBTITrig) );
    Global3DVector dir = theDTTrigger->CMSDirection( &(*iterBTITrig) );

    /// Prepare the BTI to be stored and push them into the output
    aDTBti = new DTBtiTrigger( *iterBTITrig, pos, dir );
    theBtiTrigsToStore->push_back(*aDTBti);

    /// Clean the pointer
    delete aDTBti;
  }

  /// Set the flag
  doneBTI = true;

  /// TSTheta
  std::vector< DTChambThSegm > theTSThetaTrigs = theDTTrigger->TSThTrigs();
  std::vector< DTChambThSegm >::const_iterator iterTSTheta;

  for ( iterTSTheta = theTSThetaTrigs.begin();
        iterTSTheta != theTSThetaTrigs.end();
        iterTSTheta++ )
  {
    /// Store the TSTheta trigger
    theTSThetaTrigsToStore->push_back(*iterTSTheta);
  }

  /// Set the flag
  doneTSTheta = true;

  /// TSPhi
  std::vector< DTChambPhSegm > theTSPhiTrigs = theDTTrigger->TSPhTrigs();
  std::vector< DTChambPhSegm >::const_iterator iterTSPhi;
  DTTSPhiTrigger* aTSPhiTrig;

  for ( iterTSPhi = theTSPhiTrigs.begin();
        iterTSPhi != theTSPhiTrigs.end();
        iterTSPhi++ )
  {
    /// Get the position and direction of the TSPhi trigger
    Global3DPoint pos = theDTTrigger->CMSPosition( &(*iterTSPhi) );
    Global3DVector dir = theDTTrigger->CMSDirection( &(*iterTSPhi) );

    /// Create and store the TSPhi trigger
    aTSPhiTrig = new DTTSPhiTrigger( *iterTSPhi, pos, dir );
    theTSPhiTrigsToStore->push_back(*aTSPhiTrig);

    /// Get the chamber Id
    const DTChamberId thisChambId = iterTSPhi->ChamberId();

    /// Find the Id of the two BTI's that compose the segment
    DTBtiId innerBti( thisChambId, 1, iterTSPhi->tracoTrig()->posIn() );
    DTBtiId outerBti( thisChambId, 3, iterTSPhi->tracoTrig()->posOut() );

    /// Find the matches with the BTI's
    bool foundTSPhiBtiMatch = false;

    if ( !useTSTheta && doneBTI &&
         ( iterTSPhi->station() == 1 || iterTSPhi->station() == 2 ) )
    {
      for ( BtiTrigsCollection::iterator thisBti = theBtiTrigsToStore->begin();
            thisBti != theBtiTrigsToStore->end();
            thisBti++ )
      {
        /// Add DTMatch to DTMatchesCollection.
        if( this->match( *thisBti, *iterTSPhi ) )
        {
          int tempCode = aTSPhiTrig->code() * 4;
          if ( aTSPhiTrig->station() == 1 )
          {
            tempCode += 2;
          }
          if ( thisBti->code() == 8 )
          {
            tempCode += 1;
          }

          float tempTheta = atan( sqrt( aTSPhiTrig->cmsPosition().x()*aTSPhiTrig->cmsPosition().x() + 
                                        aTSPhiTrig->cmsPosition().y()*aTSPhiTrig->cmsPosition().y() ) /
                                  aTSPhiTrig->cmsPosition().z() );

          if ( aTSPhiTrig->cmsPosition().z() < 0 )
          {
            tempTheta += M_PI;
          }

          int intTempTheta = static_cast< int >( tempTheta * 4096. );

#ifdef npDEBUG
          std::cerr << "*** CREATING MATCH from TSPHI and BTI" << std::endl;
#endif

          /// Create the DT match using position as given by TSPhi
          DTMatch* aDTMatch = new DTMatch( aTSPhiTrig->wheel(),
                                           aTSPhiTrig->station(),
                                           aTSPhiTrig->sector(),
                                           aTSPhiTrig->step(),
                                           tempCode,
                                           aTSPhiTrig->phi(),
                                           aTSPhiTrig->phiB(),
                                           intTempTheta,
                                           aTSPhiTrig->cmsPosition(),
                                           aTSPhiTrig->cmsDirection(),
                                           (aTSPhiTrig->step() == 16) ? true : false );

/*
          tempTheta = atan( sqrt( thisBti->cmsPosition().x()*thisBti->cmsPosition().x() + 
                                  thisBti->cmsPosition().y()*thisBti->cmsPosition().y() ) /
                            thisBti->cmsPosition().z() );

          if ( thisBti->cmsPosition().z() < 0 )
          {
            tempTheta += M_PI;
          }

          int intTempTheta = static_cast< int >( tempTheta * 4096. );

          /// Create the DT match using position as given by BTI
          DTMatch* aDTMatch = new DTMatch( iterTSTheta->wheel(),
                                           iterTSTheta->station(),
                                           iterTSTheta->sector(),
                                           iterTSTheta->step(),
                                           tempCode,
                                           iterTSTheta->phi(),
                                           iterTSTheta->phiB(),
                                           intTempTheta,
                                           thisBti->cmsPosition(),
                                           thisBti->cmsDirection(),
                                           (iterTSTheta->step() == 16) ? true : false );
*/

          aDTMatch->setInnerBtiId( innerBti );
          aDTMatch->setOuterBtiId( outerBti );
          aDTMatch->setMatchedBtiId( thisBti->parentId() );

          if ( aTSPhiTrig->station() == 1 || aTSPhiTrig->station() == 2 ) /// Redundant
          {
            theDTMatchContainer->at(aTSPhiTrig->station()).push_back( aDTMatch );
          }
          else
          {
            std::cerr << "A L E R T! Muon station is neither 1 nor 2" << std::endl;
          }

          /// Set the flag
          foundTSPhiBtiMatch = true;
        }
      }
    } /// End of matches TSPhi-BTI (theta)

    /// Find the matches with TSTheta's
    bool foundTSPhiTSThetaMatch = false;

    if ( useTSTheta && doneTSTheta &&
         ( iterTSPhi->station() == 1 || iterTSPhi->station() == 2 ) )
    {
      for ( TSThetaTrigsCollection::iterator thisTSTheta = theTSThetaTrigsToStore->begin();
            thisTSTheta != theTSThetaTrigsToStore->end();
            thisTSTheta++ )
      {
        /// Prepare the DTMatch
        if ( this->match( *thisTSTheta, *iterTSPhi ) )
        {
          /// Loop on TSTheta Trigger words
          for ( int i = 0; i < 7; i++ )
          {
            if ( thisTSTheta->code(i) > 0 )
            {
              /// Get the position information
              int thwh = thisTSTheta->ChamberId().wheel();
              int thstat = thisTSTheta->ChamberId().station();
              int thsect = thisTSTheta->ChamberId().sector();
              int thqual = thisTSTheta->quality(i);

              /// Identify the BTI
              int idBti = (i+1)*8 - 3;
              DTBtiId id = DTBtiId( thwh, thstat, thsect, 2, idBti );

              /// Get the chamber
              DTChamberId chaid = DTChamberId( thwh, thstat, thsect );
              const DTChamber* chamb = theMuonDTGeometryHandle->chamber(chaid);

              /// Get the chamber geometry and the BTI position
              DTTrigGeom* thisChamberGeometry = new DTTrigGeom( const_cast< DTChamber* >(chamb), false );
              GlobalPoint  posBti = thisChamberGeometry->CMSPosition( DTBtiId(chaid, 2, idBti) );
              float thposx = posBti.x();
              float thposy = posBti.y();
              float thposz = posBti.z();

              /// Get again the position information from the TSPhi
              int wh   = iterTSPhi->wheel();
              int st   = iterTSPhi->station();
              int se   = iterTSPhi->sector();
              int bx   = iterTSPhi->step();
              int code = iterTSPhi->code()*4 + thqual;

              /// Adjust the code
              /// code 0,1 = L
              ///      2,3 = H
              ///        4 = LL
              ///        5 = HL
              ///        6 = HH 
              if ( iterTSPhi->station() == 1 )
              {
                code = code + 2;
              }
              if ( code == 1 && thqual == 1 )
              {
                code = code + 1; /// Correction to avoid twice same code
              }

              /// Get and adjust polar coordinates
              int phi  = iterTSPhi->phi();
              int phib = iterTSPhi->phiB();
              float theta = atan( sqrt( thposx*thposx + thposy*thposy ) / thposz );

              if ( thposz < 0 )
              {
                theta += M_PI;
              }

              /// Set flag for trigger at right bx
              bool flagBxOK = false;
              if ( bx == 16 )
              {
                flagBxOK = true;
              }

              /// Prepare the match
              GlobalPoint posMatch( pos.x(), pos.y(), posBti.z() );

              /// Correct for asymmetries in muon LUT's
              if ( code < 16 )
              {
                if ( wh == 1 || wh == 2 ||
                     ( wh == 0 &&
                       ( se == 2 || se == 3 || se == 6 || se == 7 || se == 10 || se == 11) ) )
                {
                  /// Positive wheels
                  phib = phib - ( -17 + 4*(st-1) ) - (3-wh);
                }
                else
                {
                  /// Negative wheels
                  phib = phib - ( 11 + 4*(st-1) ) - (-3-wh);
                }
              }

              /// Store the TSPhi-TSTheta match
#ifdef npDEBUG
              std::cerr << "*** CREATING MATCH from TSPHI and TSTHETA" << std::endl;
#endif
              /// Create the DT match using position as given by the match and direction by TSPhi
              int intTempTheta = static_cast< int >( theta * 4096. );
              DTMatch* aDTMatch = new DTMatch( wh, st, se, bx, code, phi, phib, intTempTheta,
                                               posMatch, dir, flagBxOK );

              aDTMatch->setInnerBtiId( innerBti );
              aDTMatch->setOuterBtiId( outerBti );
              aDTMatch->setMatchedBtiId( id );

              if ( st == 1 || st == 2 ) /// Redundant
              {
                theDTMatchContainer->at(st).push_back( aDTMatch );
              }
              else
              {
                std::cerr << "A L E R T! Muon station is neither 1 nor 2" << std::endl;
              }

              /// Set the flag
              foundTSPhiTSThetaMatch = true;
            }
          } /// End of loop over TSTheta words to get idBti = (i+1)*8 - 3

        } /// End of if( match( *thisTSTheta, *iterTSPhi ) )
      } /// End of loop over TSTheta triggers
    } /// End of match between TSPhi and TSTheta triggers

    /// Case where no Theta Trigger was found: 
    /// this method is allowed only for TSPhi trigger code >= 2 !! 
    /// Get chamber center and allow error on whole chamber width
    if ( !(foundTSPhiBtiMatch || foundTSPhiTSThetaMatch) &&
         (iterTSPhi->station() == 1 || iterTSPhi->station() == 2) &&
         iterTSPhi->code() >= 2 )
    {
      /// Same comments as before, unless specified
      int wh   = iterTSPhi->wheel();
      int st   = iterTSPhi->station();
      int se   = iterTSPhi->sector();
      int bx   = iterTSPhi->step();
      int code = iterTSPhi->code()*4 ;
      if ( iterTSPhi->station() == 1 )
      {
        code = code + 2;
      }
      /// Force lower rank assuming it is like station 2  

      if( useRoughTheta )
      {
        int phi  = iterTSPhi->phi();
        int phib = iterTSPhi->phiB();

        /// Use the wire: theta is set at station center with half a chamber window
        DTWireId wireId = DTWireId( wh, st, se, 2, 1, 1 );
        const DTLayer* layer = theMuonDTGeometryHandle->layer( wireId.layerId() );
        const DTTopology& tp = layer->specificTopology();

        float posX = tp.wirePosition( tp.firstChannel() );
        LocalPoint posInLayer( posX, 0., 0. );
        GlobalPoint posFirstWire = layer->surface().toGlobal( posInLayer );

        int ncells = layer->specificTopology().channels();

        posX = posX + static_cast< float >(ncells) * 4.2;
        LocalPoint posInLayer2( posX, 0., 0. );
        GlobalPoint posLastWire = layer->toGlobal( posInLayer2 );

        float theta = ( posFirstWire.theta() + posLastWire.theta() )/2.;

        posX = posX - static_cast< float >(ncells) * 2.1;
        LocalPoint posInLayer3( posX, 0., 0. );
        GlobalPoint posCentralWire = layer->toGlobal( posInLayer3 );

        GlobalVector gdbti = GlobalVector(); /// Dummy direction
        bool flagBxOK = false;
        if ( bx == 16 )
        {
          flagBxOK = true;
        }

        GlobalPoint posMatch( pos.x(), pos.y(), posCentralWire.z() );

        /// Correct for asymmetries in muon LUT's
        if ( code < 16 )
        {
          if ( wh == 1 || wh == 2 ||
               ( wh == 0 &&
                 ( se == 2 || se == 3 || se == 6 || se == 7 || se == 10 || se == 11) ) )
          {
            /// Positive wheels
            phib = phib - ( -17 + 4*(st-1) ) - (3-wh);
          }
          else
          {
            /// Negative wheels
            phib = phib - ( 11 + 4*(st-1) ) - (-3-wh);
          }
        }

#ifdef npDEBUG
        std::cerr << "*** CREATING MATCH from TSPHI ONLY!!" << std::endl;
#endif

        /// Store the TSPhi-TSTheta match
        /// Create the DT match using position as given by the match and direction by BTI
        int intTempTheta = static_cast< int >( theta * 4096. );
        DTMatch* aDTMatch = new DTMatch( wh, st, se, bx, code, phi, phib, intTempTheta,
                                         posMatch, gdbti, flagBxOK );

        aDTMatch->setInnerBtiId( innerBti );
        aDTMatch->setOuterBtiId( outerBti );
        /// Leave the matched BTI Id to 0x0

        /// Set needed data for correct extrapolation search and flag for missing theta
        float deltaTheta = fabs( ( posFirstWire.theta() - posLastWire.theta() ) / sqrt(12.) ); /// Error set to sigma of flat distribution
        aDTMatch->setThetaCorrection( deltaTheta );

        if ( st == 1 || st == 2 )
        {
          theDTMatchContainer->at(st).push_back( aDTMatch );
        }
        else
        {
          std::cerr << "A L E R T! Muon station is neither 1 nor 2" << std::endl;
        }
      } // End if useRoughTheta
    } // End if no match is found ...

    /// Clean the pointer
    delete aTSPhiTrig;
  }

#ifdef npDEBUG
  std::cerr << std::endl;
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* DT TRIGGER OPERATIONS         *" << std::endl;
  std::cerr << "*********************************" << std::endl;
  std::cerr << "* done BTI? " << doneBTI << std::endl;
  std::cerr << "* done TSTheta? " << doneTSTheta << std::endl;
#endif

  return;
}

