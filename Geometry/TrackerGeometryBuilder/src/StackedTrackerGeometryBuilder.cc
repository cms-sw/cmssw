/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Unknown                              ///
///                                      ///
/// Changed by:                          ///
/// Eric Brownson                        ///
/// for compatibility with Hybrid Geom.  ///
///                                      ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2012, May, July, November            ///
///                                      ///
/// Added features:                      ///
/// PXB-PXB pairing re-written in order  ///
/// to take into account features of     ///
/// XML files produced by tkLayout       ///
/// Expanded to handle PXF-PXF pairs     ///
/// Re-defined range of different fields ///
/// to be 1 to N instead of 0 to N-1, in ///
/// a way consistent with all the other  ///
/// conventions used for DetId's
/// ////////////////////////////////////////

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometryBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include <iomanip>
#include <fstream>
#include <time.h>


StackedTrackerGeometry* StackedTrackerGeometryBuilder::build( const TrackerGeometry* theTracker,
							      double radial_window,
							      double phi_window,
							      double z_window,
							      int truncation_precision,
							      bool makeDebugFile )
{
  // For legacy compatibility it takes more inputs than it needs...
  time_t start_time = time (NULL);
  StackedTrackerGeometry* StackedTrackergeom = new StackedTrackerGeometry( theTracker );
  TrackingGeometry::DetUnitContainer::const_iterator trkIterator,trkIterator1,trkIterator2,trkIter_start,trkIter_end;

  /// Make maps to make sure things are ordered properly

  //////////
  /// BARREL
  /// Layouts from tkLayout tool will return Rods:
  /// Layer is the same, Ladder is the same, Module is different
  /// for each Pt Module

  /// Map for Layer --> Stack Layer
  std::map< uint32_t, uint32_t > layToStackMap;
  uint32_t                       nStackLayers = 0;

  /// Maps for Layer->Rod->Module->z
  std::map< uint32_t, double >                                                       *modToZMap;
  std::map< uint32_t, std::map< uint32_t, double > >                                 *rodToModToZMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, double > > >           layToRodToModToZMap;
  std::map< uint32_t, double >::iterator                                             modToZMapIter;
  std::map< uint32_t, double >::iterator                                             modToZMapIterTemp;
  std::map< uint32_t, std::map< uint32_t, double > >::iterator                       rodToModToZMapIter;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, double > > >::iterator layToRodToModToZMapIter;

  /// Maps for Layer->Rod->Module->number of z modules on one side of the current one
  /// This is to define the iZ of the Pt Module
  std::map< uint32_t, int >                                                          modToZIdxMap;
  std::map< uint32_t, std::map< uint32_t, int > >                                    rodToModToZIdxMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, int > > >              layToRodToModToZIdxMap;

  /// Maps for Layer->Rod->Module->paired Module
  std::map< uint32_t, uint32_t >                                                     modToModMap;
  std::map< uint32_t, std::map< uint32_t, uint32_t > >                               rodToModToModMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, uint32_t > > >         layToRodToModToModMap;

  /// Checksums
  std::map< uint32_t, uint32_t > rodsPerLayer;
  std::map< uint32_t, uint32_t > modsPerRod;

  //////////
  /// ENDCAP
  /// Layouts from tkLayout wil return Rings:
  /// Disk is the same, Ring is in fact encoded as a Blade
  /// Ring-Blade is the same, Panel is always 1, Module is different
  /// for each Pt Module

  /// Map for Disk --> Stack Disk
  std::map< uint32_t, uint32_t > diskToStackMap;
  uint32_t                       nStackDisks = 0;

  /// Maps for Side->Disk->Ring/Blade->Module->phi
  std::map< uint32_t, double >                                                                             *modToPhiMap;
  std::map< uint32_t, std::map< uint32_t, double > >                                                       *ringToModToPhiMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, double > > >                                 *diskToRingToModToPhiMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, double > > > >           sideToDiskToRingToModToPhiMap;
  std::map< uint32_t, double >::iterator                                                                   modToPhiMapIter;
  std::map< uint32_t, double >::iterator                                                                   modToPhiMapIterTemp;
  std::map< uint32_t, std::map< uint32_t, double > >::iterator                                             ringToModToPhiMapIter;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, double > > >::iterator                       diskToRingToModToPhiMapIter;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, double > > > >::iterator sideToDiskToRingToModToPhiMapIter;

  /// Maps for Side->Disk->Ring/Blade->Module->number of phi modules on one side of the current one
  /// This is to define the iPhi of the Pt Module
  std::map< uint32_t, int >                                                                                modToPhiIdxMap;
  std::map< uint32_t, std::map< uint32_t, int > >                                                          ringToModToPhiIdxMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, int > > >                                    diskToRingToModToPhiIdxMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, int > > > >              sideToDiskToRingToModToPhiIdxMap;

  /// Maps for Side->Disk->Ring/Blade->Module->paired Module
  /// modToModMap already declared for BARREL:
  /// it should be safe to use the same container
  /// as it is destroyed at each loop ...
  std::map< uint32_t, std::map< uint32_t, uint32_t > >                                                     ringToModToModMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, uint32_t > > >                               diskToRingToModToModMap;
  std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, uint32_t > > > >         sideToDiskToRingToModToModMap;

  /// Checksums
  std::map< uint32_t, uint32_t >                       ringsPerDisk;
  std::map< uint32_t, std::map< uint32_t, uint32_t > > modsPerRingPerDisk;

  { /// Destroy temp variables later
    /// First store the layer, module z's in a map
    bool firstElement = true;
    for ( trkIterator = theTracker->detUnits().begin();
          trkIterator != theTracker->detUnits().end();
          ++trkIterator )
    {
      DetId id = (**trkIterator).geographicalId();
      double r = (**trkIterator).position().perp();
      double z = (**trkIterator).position().z();

      ///////////////////
      /// Map PixelBarrel
      if ( (**trkIterator).type().isBarrel() &&
           (**trkIterator).type().isTrackerPixel() &&
           (r>20.0) )
      {
        if (firstElement)
        {
          trkIter_start = trkIterator;
          firstElement = false;
        }
        trkIter_end = trkIterator; /// Define subset of theTracker to run over later for speed

        uint32_t lay = PXBDetId(id).layer();
        uint32_t rod = PXBDetId(id).ladder();
        uint32_t mod = PXBDetId(id).module();

        /// These are for later checksums
        if ( rodsPerLayer.find(lay) == rodsPerLayer.end() )
          rodsPerLayer.insert( std::make_pair(lay, 0) );
        if ( modsPerRod.find(lay) == modsPerRod.end() )
          modsPerRod.insert( std::make_pair(lay, 0) );
        if ( rod > rodsPerLayer[lay] )
          rodsPerLayer[lay] = rod;
        if ( mod > modsPerRod[lay] )
          modsPerRod[lay] = mod;

        double zModule = (**trkIterator).position().z();

        /// Fill Maps
        layToRodToModToZMapIter = layToRodToModToZMap.find(lay);
        if (layToRodToModToZMapIter != layToRodToModToZMap.end())
        {
          /// Layer found!
          rodToModToZMap = &layToRodToModToZMapIter->second;
          rodToModToZMapIter = rodToModToZMap->find(rod);
          if (rodToModToZMapIter != rodToModToZMap->end())
          {
            /// Rod found!
            modToZMap = &rodToModToZMapIter->second;
            modToZMapIter = modToZMap->find(mod);
            if (modToZMapIter != modToZMap->end())
            {
              /// Module found!
              std::cerr << "A L E R T! Layer-Rod-Module already present!" << std::endl;
            }
            else
            {
              /// New module!
              modToZMap->insert( std::make_pair( mod, zModule ) );
            }
          }
          else
          {
            /// New rod!
            /// New module!
            std::map< uint32_t, double > tempMap;
            tempMap.insert( std::make_pair( mod, zModule ) );
            rodToModToZMap->insert( std::make_pair( rod, tempMap ) );
          }
        }
        else
        {
          /// New Layer!
          /// New rod!
          /// New module!
          std::map< uint32_t, double > tempMap1;
          tempMap1.insert( std::make_pair( mod, zModule ) );
          std::map< uint32_t, std::map< uint32_t, double > > tempMap2;
          tempMap2.insert( std::make_pair( rod, tempMap1 ) );
          layToRodToModToZMap.insert( std::make_pair( lay, tempMap2 ) );

          if (layToStackMap.find(lay) == layToStackMap.end())
          {
            layToStackMap.insert( std::make_pair( lay , nStackLayers ) );     // assumes that the layers are ordered already
            nStackLayers++;
          }
        }
      }

      ////////////////////
      /// Map PixelForward
      if ( (**trkIterator).type().isEndcap() &&
           (**trkIterator).type().isTrackerPixel() &&
           (fabs(z)>70.0) )
      {
        if (firstElement)
        {
          trkIter_start = trkIterator;
          firstElement = false;
        }
        trkIter_end = trkIterator; /// Define subset of theTracker to run over later for speed

        uint32_t side = PXFDetId(id).side();
        uint32_t disk = PXFDetId(id).disk();
        uint32_t ring = PXFDetId(id).ring();
        uint32_t mod  = PXFDetId(id).module();

        /// These are for later checksums
        if ( ringsPerDisk.find(disk) == ringsPerDisk.end() )
          ringsPerDisk.insert( std::make_pair(disk, 0) );
        if ( modsPerRingPerDisk.find(disk) == modsPerRingPerDisk.end() )
        {
          std::map< uint32_t, uint32_t > tempMap;
          tempMap.insert( std::make_pair(ring, 0) );
          modsPerRingPerDisk.insert( std::make_pair(disk, tempMap) );
        }
        if ( modsPerRingPerDisk.find(disk)->second.find(ring) == modsPerRingPerDisk.find(disk)->second.end() )
          modsPerRingPerDisk.find(disk)->second.insert( std::make_pair( ring, 0 ) );
        if ( ring > ringsPerDisk[disk] )
          ringsPerDisk[disk] = ring;
        if ( mod > modsPerRingPerDisk.find(disk)->second.find(ring)->second )
          modsPerRingPerDisk.find(disk)->second.find(ring)->second = mod;

        double phiModule = (**trkIterator).position().phi();
        if (phiModule < 0) phiModule += 2*M_PI;

        /// Fill Maps
        sideToDiskToRingToModToPhiMapIter = sideToDiskToRingToModToPhiMap.find(side);
        if (sideToDiskToRingToModToPhiMapIter != sideToDiskToRingToModToPhiMap.end())
        {
          /// Side found!
          diskToRingToModToPhiMap = &sideToDiskToRingToModToPhiMapIter->second;
          diskToRingToModToPhiMapIter = diskToRingToModToPhiMap->find(disk);
          if (diskToRingToModToPhiMapIter != diskToRingToModToPhiMap->end())
          {
            /// Disk found!
            ringToModToPhiMap = &diskToRingToModToPhiMapIter->second;
            ringToModToPhiMapIter = ringToModToPhiMap->find(ring);
            if (ringToModToPhiMapIter != ringToModToPhiMap->end())
            {
              /// Ring found!
              modToPhiMap = &ringToModToPhiMapIter->second;
              modToPhiMapIter = modToPhiMap->find(mod);
              if (modToPhiMapIter != modToPhiMap->end())
              {
                /// Module found!
                std::cerr << "A L E R T! Side-Disk-Ring-Module already present!" << std::endl;
              }
              else
              {
                /// New module!
                modToPhiMap->insert( std::make_pair( mod, phiModule ) );
              }
            }
            else
            {
              /// New ring!
              /// New module!
              std::map< uint32_t, double > tempMap;
              tempMap.insert( std::make_pair( mod, phiModule ) );
              ringToModToPhiMap->insert( std::make_pair( ring, tempMap ) );
            }
          }
          else
          {
            /// New disk!
            /// New ring!
            /// New module!
            std::map< uint32_t, double > tempMap1;
            tempMap1.insert( std::make_pair( mod, phiModule ) );
            std::map< uint32_t, std::map< uint32_t, double > > tempMap2;
            tempMap2.insert( std::make_pair( ring, tempMap1 ) );
            diskToRingToModToPhiMap->insert( std::make_pair( disk, tempMap2 ) );

            if (diskToStackMap.find(disk) == diskToStackMap.end())
            {
              diskToStackMap.insert( std::make_pair( disk , nStackDisks ) );     // assumes that the disks are ordered already and FW/BW symmetric
              nStackDisks++;
            }
          }
        }
        else
        {
          /// New side!
          /// New disk!
          /// New ring!
          /// New module!
          std::map< uint32_t, double > tempMap1;
          tempMap1.insert( std::make_pair( mod, phiModule ) );
          std::map< uint32_t, std::map< uint32_t, double > > tempMap2;
          tempMap2.insert( std::make_pair( ring, tempMap1 ) );
          std::map< uint32_t, std::map< uint32_t, std::map< uint32_t, double > > > tempMap3;
          tempMap3.insert( std::make_pair( disk, tempMap2 ) );
          sideToDiskToRingToModToPhiMap.insert( std::make_pair( side, tempMap3 ) );

          if (diskToStackMap.find(disk) == diskToStackMap.end())
          {
            diskToStackMap.insert( std::make_pair( disk , nStackDisks ) );     // assumes that the disks are ordered already and FW/BW symmetric
            nStackDisks++;
          }

        }
      }
      /// End of Map of PixelBarrel, PixelForward
      ///////////////////////////////////////////

    }
    trkIter_end++;

    //////////
    /// BARREL
    /// Find the closest module in z
    for ( layToRodToModToZMapIter = layToRodToModToZMap.begin();
          layToRodToModToZMapIter != layToRodToModToZMap.end();
          ++layToRodToModToZMapIter )
    {
      rodToModToZMap = &layToRodToModToZMapIter->second;
      rodToModToModMap.clear();
      rodToModToZIdxMap.clear();

      for ( rodToModToZMapIter = (*rodToModToZMap).begin();
            rodToModToZMapIter != (*rodToModToZMap).end();
            ++rodToModToZMapIter )
      {
        modToZMap = &rodToModToZMapIter->second;
        modToModMap.clear();
        modToZIdxMap.clear();

        for ( modToZMapIter = (*modToZMap).begin();
              modToZMapIter != (*modToZMap).end();
              ++modToZMapIter )
        {
          double tempZ = modToZMapIter->second;
          double deltaZ = 9999999.9;
          uint32_t closeMod = 0xFFFFFFFF;
          uint32_t smallerz = 0;

          /// Find closest Z
          for ( modToZMapIterTemp = (*modToZMap).begin();
                modToZMapIterTemp != (*modToZMap).end();
                ++modToZMapIterTemp )
          {
            if ( modToZMapIter->first == modToZMapIterTemp->first ) continue;

            double closeZ = modToZMapIterTemp->second;
            if ( fabs(closeZ - tempZ) < deltaZ )
            {
              deltaZ = fabs(closeZ - tempZ);
              closeMod = modToZMapIterTemp->first;
            }

            if ( tempZ > closeZ ) smallerz++;
          }

          /// Here we have the closest module
          /// stored in closeMod
          modToModMap.insert( std::make_pair( modToZMapIter->first, closeMod ) );
          modToZIdxMap.insert( std::make_pair( modToZMapIter->first, smallerz ) );
        }
        rodToModToModMap.insert( std::make_pair( rodToModToZMapIter->first, modToModMap ) );
        rodToModToZIdxMap.insert( std::make_pair( rodToModToZMapIter->first, modToZIdxMap ) );
      }
      layToRodToModToModMap.insert( std::make_pair( layToRodToModToZMapIter->first, rodToModToModMap ) );
      layToRodToModToZIdxMap.insert( std::make_pair( layToRodToModToZMapIter->first, rodToModToZIdxMap ) );
    }

    //////////
    /// ENDCAP
    /// Find the closest module in phi
    for ( sideToDiskToRingToModToPhiMapIter = sideToDiskToRingToModToPhiMap.begin();
          sideToDiskToRingToModToPhiMapIter != sideToDiskToRingToModToPhiMap.end();
          ++sideToDiskToRingToModToPhiMapIter )
    {
      diskToRingToModToPhiMap = &sideToDiskToRingToModToPhiMapIter->second;
      diskToRingToModToModMap.clear();
      diskToRingToModToPhiIdxMap.clear();

      for ( diskToRingToModToPhiMapIter = (*diskToRingToModToPhiMap).begin();
            diskToRingToModToPhiMapIter != (*diskToRingToModToPhiMap).end();
            ++diskToRingToModToPhiMapIter )
      {
        ringToModToPhiMap = &diskToRingToModToPhiMapIter->second;
        ringToModToModMap.clear();
        ringToModToPhiIdxMap.clear();

        for ( ringToModToPhiMapIter = (*ringToModToPhiMap).begin();
              ringToModToPhiMapIter != (*ringToModToPhiMap).end();
              ++ringToModToPhiMapIter )
        {
          modToPhiMap = &ringToModToPhiMapIter->second;
          modToModMap.clear();
          modToPhiIdxMap.clear();

          for ( modToPhiMapIter = (*modToPhiMap).begin();
                modToPhiMapIter != (*modToPhiMap).end();
                ++modToPhiMapIter )
          {
            double tempPhi = modToPhiMapIter->second;
            double deltaPhi = 9999999.9;
            uint32_t closeMod = 0xFFFFFFFF;
            uint32_t smallerphi = 0;

            /// Find closest Phi
            for ( modToPhiMapIterTemp = (*modToPhiMap).begin();
                  modToPhiMapIterTemp != (*modToPhiMap).end();
                  ++modToPhiMapIterTemp )
            {
              if ( modToPhiMapIter->first == modToPhiMapIterTemp->first ) continue;

              double closePhi = modToPhiMapIterTemp->second;
              double tempDeltaPhi = closePhi - tempPhi;
              if ( tempDeltaPhi < 0 ) tempDeltaPhi = -tempDeltaPhi;
              if ( tempDeltaPhi > M_PI ) tempDeltaPhi = 2*M_PI - tempDeltaPhi;
              if ( fabs(tempDeltaPhi) < deltaPhi )
              {
                deltaPhi = fabs(tempDeltaPhi);
                closeMod = modToPhiMapIterTemp->first;
              }

              if ( tempPhi > closePhi ) smallerphi++;
            }

            /// Here we have the closest module
            /// stored in closeMod
            modToModMap.insert( std::make_pair( modToPhiMapIter->first, closeMod ) );
            modToPhiIdxMap.insert( std::make_pair( modToPhiMapIter->first, smallerphi ) );
          }
          ringToModToModMap.insert( std::make_pair( ringToModToPhiMapIter->first, modToModMap ) );
          ringToModToPhiIdxMap.insert( std::make_pair( ringToModToPhiMapIter->first, modToPhiIdxMap ) );
        }
        diskToRingToModToModMap.insert( std::make_pair( diskToRingToModToPhiMapIter->first, ringToModToModMap ) );
        diskToRingToModToPhiIdxMap.insert( std::make_pair( diskToRingToModToPhiMapIter->first, ringToModToPhiIdxMap ) );
      }
      sideToDiskToRingToModToModMap.insert( std::make_pair( sideToDiskToRingToModToPhiMapIter->first, diskToRingToModToModMap ) );
      sideToDiskToRingToModToPhiIdxMap.insert( std::make_pair( sideToDiskToRingToModToPhiMapIter->first, diskToRingToModToPhiIdxMap ) );
    }

  } /// Destroy temp variables with this bracket

  /// Perform the matching
  std::map< uint32_t, uint32_t > detIdToDetIdMap;

  std::map< uint32_t, uint32_t > counterB;
  std::map< uint32_t, uint32_t > counterE;
  bool fastExit = false;
  for ( trkIterator1 = trkIter_start;
        trkIterator1 != trkIter_end;
        ++trkIterator1 )
  {
    DetId id1 = (**trkIterator1).geographicalId();
    double r1 = (**trkIterator1).position().perp();
    double z1 = (**trkIterator1).position().z();

    /// Match PixelBarrel to PixelBarrel
    if ( (**trkIterator1).type().isBarrel() &&
         (**trkIterator1).type().isTrackerPixel() &&
         (r1>20.0) &&
         detIdToDetIdMap.find(id1) == detIdToDetIdMap.end() )
    {
      uint32_t lay1 = PXBDetId(id1).layer();
      uint32_t rod1 = PXBDetId(id1).ladder();
      uint32_t mod1 = PXBDetId(id1).module();

      /// Nested loop
      fastExit = false;
      for ( trkIterator2 = trkIter_start;
            trkIterator2 != trkIter_end && !fastExit;
            ++trkIterator2 )
      {
        DetId id2 = (**trkIterator2).geographicalId();
        double r2 = (**trkIterator2).position().perp();

        if ( (**trkIterator2).type().isBarrel() &&
             (**trkIterator2).type().isTrackerPixel() &&
             (r2>20.0) )
        {
          uint32_t lay2 = PXBDetId(id2).layer();
          uint32_t rod2 = PXBDetId(id2).ladder();
          uint32_t mod2 = PXBDetId(id2).module();

          /// Matching conditions
          if (lay2 != lay1) continue;
          if (rod2 != rod1) continue;
          if (mod2 <= mod1) continue; /// To avoid duplicates (since both pairs 1-2 and 2-1 are accepted)

          /// Get the matched module!
          rodToModToModMap = layToRodToModToModMap.find(lay1)->second;
          modToModMap = rodToModToModMap.find(rod1)->second;
          uint32_t tempMod1 = modToModMap.find(mod1)->second;
          if (mod2 != tempMod1) continue;

          /// Here we have same layer, same rod, paired modules
          StackedTrackerDetUnit::StackContents listStackMembers ;
          if (r1 < r2)
          {
            listStackMembers.insert( std::make_pair( 0 , id1 ) );
            listStackMembers.insert( std::make_pair( 1 , id2 ) );
          } // first one should be the inner sensor
          else if (r1 > r2)
          {
            listStackMembers.insert( std::make_pair( 0 , id2 ) );
            listStackMembers.insert( std::make_pair( 1 , id1 ) );
          } // first one should be the inner sensor
          else
            throw cms::Exception("StackedTrackerGeometryBuilder") << "E R R O R! modules coincide! "
                                                                  << mod1 << " " << mod2 << " [ located at... ] "
                                                                  << (**trkIterator1).position().z() << " " << (**trkIterator2).position().z() << std::endl;

          uint32_t tempMod2 = layToRodToModToZIdxMap.find(lay1)->second.find(rod1)->second.find(mod1)->second;
          /// StackedTrackerDetId aStackId( uint32_t(layToStackMap.find(lay1)->second), rod1-1, tempMod2/2 );
          /// Renormalize ROD from 0 to N-1 instead of from 1 to N!
          /// NOTE: this was the old way
          /// starting from 6_X_Y the usual convention of 0 = wildcard
          /// for DetId's is applied also here!
          StackedTrackerDetId aStackId( uint32_t(layToStackMap.find(lay1)->second)+1, rod1, tempMod2/2+1 );

          /// A few checks to find possible errors...
          /// Duplicate Stacks
          if ( StackedTrackergeom->idToStack(aStackId) != NULL )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build a duplicate ID from Barrel:" << aStackId << std::endl;

          /// DeltaPhi
          double phi1 = (**trkIterator1).position().phi();
          double phi2 = (**trkIterator2).position().phi();
          if ( phi1 < 0.0 ) phi1 +=2.0*M_PI;
          if ( phi2 < 0.0 ) phi2 +=2.0*M_PI;
          double Dphi = fabs(phi2-phi1);
          if ( Dphi >= 2.0*M_PI ) Dphi -= 2.0*M_PI;

          /// DeltaZ
          double Dz = fabs( (**trkIterator2).position().z()-(**trkIterator1).position().z() );

          /// If you KNOW it is being built correctly you can increase your windows to allow such values.
          if ( fabs(r1-r2) > radial_window )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build Barrel stacks that are far apart in R:" << fabs(r1-r2) << " " << aStackId << std::endl;
          if ( Dphi>=phi_window )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build Barrel stacks that are far apart in phi:" << Dphi << " " << aStackId << std::endl;
          if ( Dz>=z_window )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build Barrel stacks that are far apart in Z:" << Dz << " " <<  aStackId << std::endl;

          /// If the Stack is correctly built, it can be added
          StackedTrackergeom->addStack( new StackedTrackerDetUnit(aStackId, listStackMembers) );
          detIdToDetIdMap.insert( std::make_pair(id2, id1) ); // Reverse order since the first one is being checked now

          if ( counterB.find(layToStackMap.find(lay1)->second) == counterB.end() )
            counterB.insert( std::make_pair(layToStackMap.find(lay1)->second, 0 ) );

          counterB[layToStackMap.find(lay1)->second]++;
          fastExit = true;

        }
      } /// Nested loop
    }

    /// Match PixelEndcap to PixelEndcap
    if ( (**trkIterator1).type().isEndcap() &&
         (**trkIterator1).type().isTrackerPixel() &&
         (fabs(z1)>70.0) &&
         detIdToDetIdMap.find(id1) == detIdToDetIdMap.end() )
    {
      uint32_t side1 = PXFDetId(id1).side();
      uint32_t disk1 = PXFDetId(id1).disk();
      uint32_t ring1 = PXFDetId(id1).ring();
      uint32_t mod1  = PXFDetId(id1).module();

      /// Nested loop
      fastExit = false;
      for ( trkIterator2 = trkIter_start;
            trkIterator2 != trkIter_end && !fastExit;
            ++trkIterator2 )
      {
        DetId id2 = (**trkIterator2).geographicalId();
        double z2 = (**trkIterator2).position().z();

        if ( (**trkIterator2).type().isEndcap() &&
             (**trkIterator2).type().isTrackerPixel() &&
             (fabs(z2)>70.0) )
        {
          uint32_t side2 = PXFDetId(id2).side();
          uint32_t disk2 = PXFDetId(id2).disk();
          uint32_t ring2 = PXFDetId(id2).ring();
          uint32_t mod2 = PXFDetId(id2).module();

          /// Matching conditions
          if (side1 != side2) continue;
          if (disk1 != disk2) continue;
          if (ring1 != ring2) continue;
          if (mod2 <= mod1)   continue; /// To avoid duplicates (since both pairs 1-2 and 2-1 are accepted)

          /// Get the matched module!
          diskToRingToModToModMap = sideToDiskToRingToModToModMap.find(side1)->second;
          ringToModToModMap = diskToRingToModToModMap.find(disk1)->second;
          modToModMap = ringToModToModMap.find(ring1)->second;
          uint32_t tempMod1 = modToModMap.find(mod1)->second;
          if (mod2 != tempMod1) continue;

          /// Here we have same side, same disk, same ring and paired modules
          StackedTrackerDetUnit::StackContents listStackMembers ;
          if (fabs(z1) < fabs(z2))
          {
            listStackMembers.insert( std::make_pair( 0 , id1 ) );
            listStackMembers.insert( std::make_pair( 1 , id2 ) );
          } // first one should be the inner sensor
          else if (fabs(z1) > fabs(z2))
          {
            listStackMembers.insert( std::make_pair( 0 , id2 ) );
            listStackMembers.insert( std::make_pair( 1 , id1 ) );
          } // first one should be the inner sensor
          else
            throw cms::Exception("StackedTrackerGeometryBuilder") << "E R R O R! modules coincide! "
                                                                  << mod1 << " " << mod2 << " [ located at... ] "
                                                                  << z1 << " " << z2 << std::endl;

          uint32_t tempMod2 = sideToDiskToRingToModToPhiIdxMap.find(side1)->second.find(disk1)->second.find(ring1)->second.find(mod1)->second;
          /// StackedTrackerDetId aStackId( side1, uint32_t(diskToStackMap.find(disk1)->second), ring1-1, tempMod2/2 );
          /// Renormalize RING from 0 to N-1 instead of from 1 to N!
          /// NOTE: this was the old way
          /// starting from 6_X_Y the usual convention of 0 = wildcard
          /// for DetId's is applied also here!
          StackedTrackerDetId aStackId( side1, uint32_t(diskToStackMap.find(disk1)->second)+1, ring1, tempMod2/2+1 );

          /// A few checks to find possible errors...
          /// Duplicate Stacks
          if ( StackedTrackergeom->idToStack(aStackId) != NULL )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build a duplicate ID from Endcap:" << aStackId << std::endl;

          /// DeltaPhi
          double phi1 = (**trkIterator1).position().phi();
          double phi2 = (**trkIterator2).position().phi();
          if ( phi1 < 0.0 ) phi1 +=2.0*M_PI;
          if ( phi2 < 0.0 ) phi2 +=2.0*M_PI;
          double Dphi = fabs(phi2-phi1);
          if ( Dphi >= 2.0*M_PI ) Dphi -= 2.0*M_PI;

          /// DeltaR
          double DR = fabs( (**trkIterator2).position().perp()-(**trkIterator1).position().perp() );

          /// If you KNOW it is being built correctly you can increase your windows to allow such values.
          if ( DR > radial_window )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build Endcap stacks that are far apart in R:" << DR << " " << aStackId << std::endl;
          if ( Dphi>=phi_window )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build Endcap stacks that are far apart in phi:" << Dphi << " " << aStackId << std::endl;
          if ( fabs(z1-z2)>=z_window )
            throw cms::Exception("StackedTrackerGeometryBuilder") << "Attempted to build Endcap stacks that are far apart in Z:" << fabs(z1-z2) << " " <<  aStackId << std::endl;

          /// If the Stack is correctly built, it can be added
          StackedTrackergeom->addStack( new StackedTrackerDetUnit(aStackId, listStackMembers) );
          detIdToDetIdMap.insert( std::make_pair(id2, id1) ); // Reverse order since the first one is being checked now

          if ( counterE.find(diskToStackMap.find(disk1)->second) == counterE.end() )
            counterE.insert( std::make_pair(diskToStackMap.find(disk1)->second, 0 ) );

          counterE[diskToStackMap.find(disk1)->second]++;
          fastExit = true;

        }
      } /// Nested loop
    }

  } /// End of loop over detectors

  /// Now for a few check sums and outputs
  time_t end_time = time (NULL);
  std::cout << "Found:" << std::endl;
  std::cout << std::endl;

  for ( std::map< uint32_t, uint32_t >::iterator layIterator = layToStackMap.begin();
        layIterator != layToStackMap.end();
        ++layIterator )
  {
    std::cout << "\tBarrel Stack layer " << layIterator->first << " : " << counterB[layIterator->second] << " stacks" << std::endl;
  }

  for ( std::map< uint32_t, uint32_t >::iterator layIterator = layToStackMap.begin();
        layIterator != layToStackMap.end();
        ++layIterator )
  {
    uint32_t targetnum = ( modsPerRod[layIterator->first] * rodsPerLayer[layIterator->first] )/2;
    if ( counterB[layIterator->second] != targetnum )
    {
      throw cms::Exception("StackedTrackerGeometryBuilder")
        << "There should be more Stacks in PXB layer " << layIterator->first
        << ", "<<counterB[ layIterator->second ]
        << " Stacks were found and there should be "
        << targetnum << std::endl;
    }
  }

  std::cout << std::endl;

  for ( std::map< uint32_t, uint32_t >::iterator diskIterator = diskToStackMap.begin();
        diskIterator != diskToStackMap.end();
        ++diskIterator )
  {
    std::cout << "\tEndcap Stack disk " << diskIterator->first << " : " << counterE[diskIterator->second] << " stacks" << std::endl;
  }

  for ( std::map< uint32_t, uint32_t >::iterator diskIterator = diskToStackMap.begin();
        diskIterator != diskToStackMap.end();
        ++diskIterator )
  {
    uint32_t targetnum = 0;

    std::map< uint32_t, uint32_t > tempMap;
    tempMap = modsPerRingPerDisk[diskIterator->first];
    for ( std::map< uint32_t, uint32_t >::iterator iter = tempMap.begin();
          iter != tempMap.end();
          ++iter )
      targetnum += iter->second;

    //targetnum = targetnum/2;

    if ( counterE[diskIterator->second] != targetnum )
    {
      throw cms::Exception("StackedTrackerGeometryBuilder")
        << "There should be more Stacks in PXF disk " << diskIterator->first
        << ", "<<counterE[ diskIterator->second ]
        << " Stacks were found and there should be "
        << targetnum << std::endl;
    }
  }

  std::cout << "Created " << StackedTrackergeom->stacks().size() << " stacks in " << end_time-start_time << " s" << std::endl;

  return StackedTrackergeom;
}



