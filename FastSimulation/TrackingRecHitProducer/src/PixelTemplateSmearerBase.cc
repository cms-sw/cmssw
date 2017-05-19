
/** PixelTemplateSmearerBase.cc
 * ---------------------------------------------------------------------
 * Base class for FastSim plugins to simulate all simHits on one DetUnit.
 * 
 * Petar Maksimovic (JHU), based the code by 
 * Guofan Hu (JHU) from SiPixelGaussianSmearingRecHitConverterAlgorithm.cc
 * Alice Sady (JHU): new pixel resolutions (2015) and hit merging code.
 * ---------------------------------------------------------------------
 */

// SiPixel Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/PixelTemplateSmearerBase.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"

// Famos
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/Utilities/interface/SimpleHistogramGenerator.h"

// STL

// ROOT
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>


const double microntocm = 0.0001;


PixelTemplateSmearerBase::PixelTemplateSmearerBase(
    const std::string& name,
    const edm::ParameterSet& config,
    edm::ConsumesCollector& consumesCollector 
):
    TrackingRecHitAlgorithm(name,config,consumesCollector)
{
    mergeHitsOn = config.getParameter<bool>("MergeHitsOn");
    templateId = config.getParameter<int> ( "templateId" );
}


PixelTemplateSmearerBase::~PixelTemplateSmearerBase()
{
    for (auto it = theXHistos.begin(); it != theXHistos.end(); ++it )
    {
        delete it->second;
    }
    for (auto it = theYHistos.begin(); it != theYHistos.end(); ++it )
    {
        delete it->second;
    }
    theXHistos.clear();
    theYHistos.clear();

}

TrackingRecHitProductPtr 
PixelTemplateSmearerBase::process(TrackingRecHitProductPtr product) const
{
    std::vector<std::pair<unsigned int,const PSimHit*>> & simHitIdPairs = product->getSimHitIdPairs();
    std::vector<const PSimHit*> simHits(simHitIdPairs.size());
    for (unsigned int ihit = 0; ihit<simHitIdPairs.size();++ihit)
    {
        simHits[ihit]=simHitIdPairs[ihit].second;
    }

    RandomEngineAndDistribution const & randomEngine = getRandomEngine();


    const GeomDet* geomDet = getTrackerGeometry().idToDetUnit(product->getDetId());
    const PixelGeomDetUnit * pixelGeomDet = dynamic_cast< const PixelGeomDetUnit* >( geomDet );
    if (pixelGeomDet == 0)
    {
        throw cms::Exception("FastSimulation/TrackingRecHitProducer") << "The GeomDetUnit is not a PixelGeomDetUnit.  This should never happen!";
    }
    const BoundPlane& theDetPlane = pixelGeomDet->surface();
    const Bounds& theBounds = theDetPlane.bounds();
    const double boundX = theBounds.width()/2.;
    const double boundY = theBounds.length()/2.;


    std::vector<TrackingRecHitProduct::SimHitIdPair> listOfUnmergedHits; 
    std::vector< MergeGroup* > listOfMergeGroups;
    int nHits = simHits.size();          
    
    // fixed size array, 0 if hit is unmerged             
    MergeGroup* mergeGroupByHit[nHits];             


    if (nHits == 0)
    {
        return product;
    }
    else if (nHits == 1)
    {
        listOfUnmergedHits.push_back(simHitIdPairs[0]);
    }
    else 
    {
        if ( mergeHitsOn )
        {
   
            for (int i = 0; i < nHits; ++i )
            {
                //initialize this cell to a NULL pointer here
                mergeGroupByHit[i] = nullptr;       
            }
            for ( int i = 0; i < nHits-1; ++i )
            {
                for ( int j = i+1 ; j < nHits; ++j ) {

                    //--- Calculate the distance between hits i and j:
                    bool merged = hitsMerge( *simHitIdPairs[i].second, *simHitIdPairs[j].second );


                    if ( merged )
                    {
                        // First, check if the other guy (j) is in some merge group already
                        if ( mergeGroupByHit[j] != 0 ) 
                        {
                            if (mergeGroupByHit[i] == 0 ) 
                            {
                                mergeGroupByHit[i] = mergeGroupByHit[j];
                                mergeGroupByHit[i]->group.push_back(simHitIdPairs[i]);
                                mergeGroupByHit[i]->smearIt = 1;
                            }
                            else
                            {
                                if (mergeGroupByHit[i] != mergeGroupByHit[j])
                                {
                                    for (auto hit_it = mergeGroupByHit[j]->group.begin(); hit_it != mergeGroupByHit[j]->group.end(); ++hit_it)
                                    {
                                        mergeGroupByHit[i]->group.push_back( *hit_it );
                                        mergeGroupByHit[i]->smearIt = 1;
                                    }

                                    // Step 2: iterate over all hits, replace mgbh[j] by mgbh[i] (so that nobody points to i)                               
                                    MergeGroup * mgbhj = mergeGroupByHit[j];                                                               
                                    for ( int k = 0; k < nHits; ++k )
                                    {
                                            if ( mgbhj == mergeGroupByHit[k])
                                            {
                                                // Hit k also uses the same merge group, tell them to switch to mgbh[i]                                             
                                                mergeGroupByHit[k] = mergeGroupByHit[i];
					    }
                                    }
                                    mgbhj->smearIt = 0;
                                    mergeGroupByHit[i]->smearIt = 1;

                                    //  Step 3 would have been to delete mgbh[j]... however, we'll do that at the end anyway.                              
                                    //  The key was to prevent mgbh[j] from being accessed further, and we have done that,                                 
                                    //  since now no mergeGroupByHit[] points to mgbhj any more.  Note that the above loop                                
                                    //  also set mergeGroupByHit[i] = mergeGroupByHit[j], too. 
                                }
                            }
                        }
                        else 
                        { 
                            // j is not merged.  Check if i is merged with another hit yet.
                            //
                            if ( mergeGroupByHit[i] == 0 )
                            {
                                // This is the first time we realized i is merged with any
                                // other hit.  Create a new merge group for i and j
                                mergeGroupByHit[i] = new MergeGroup();
                                listOfMergeGroups.push_back( mergeGroupByHit[i] );   // keep track of it
                                //std::cout << "new merge group " << mergeGroupByHit[i] << std::endl;
                                //
                                // Add hit i as the first to its own merge group
                                // (simHits[i] is a const pointer to PSimHit).
                                //std::cout << "ALICE: simHits" << simHits[i] << std::endl;
                                mergeGroupByHit[i]->group.push_back( simHitIdPairs[i] );
                                mergeGroupByHit[i]->smearIt = 1;
                            }
                            //--- Add hit j as well
                            mergeGroupByHit[i]->group.push_back( simHitIdPairs[j] );
                            mergeGroupByHit[i]->smearIt = 1;
                            
                            mergeGroupByHit[j] = mergeGroupByHit[i];

                        } // --- end of else if ( j has merge group )

                    } //--- end of if (merged)

                } //--- end of loop over j

                //--- At this point, there are two possibilities.  Either hit i
                //    was already chosen to be merged with some hit prior to it,
                //    or the loop over j found another merged hit.  In either
                //    case, if mergeGroupByHit[i] is empty, then the hit is
                //    unmerged.
                //
                if ( mergeGroupByHit[i] == 0 )
                {
                    //--- Keep track of it.
                    listOfUnmergedHits.push_back( simHitIdPairs[i] );
                }
            } //--- end of loop over i
        } // --- end of if (mergeHitsOn)
        else
        {
            //      std::cout << "Merged Hits Off!" << std::endl;
            //Now we've turned off hit merging, so all hits should be pushed
            //back to listOfUnmergedHits
            for (int i = 0; i < nHits; ++i )
            {
                listOfUnmergedHits.push_back( simHitIdPairs[i] );
            }
        }
    } // --- end of if (nHits == 1) else {...}


    //--- We now have two lists: a list of hits that are unmerged, and
    //    the list of merge groups.  Process each separately.
    //
    product = processUnmergedHits( listOfUnmergedHits, product,
    pixelGeomDet, boundX, boundY,
    &randomEngine );


    product = processMergeGroups(  listOfMergeGroups,  product,
    pixelGeomDet, boundX, boundY,
    &randomEngine );

    //--- We're done with this det unit, and ought to clean up used
    //    memory.  We don't own the PSimHits, and the vector of
    //    listOfUnmergedHits simply goes out of scope.  However, we
    //    created the MergeGroups and thus we need to get rid of them.
    //
    for (auto mg_it = listOfMergeGroups.begin(); mg_it != listOfMergeGroups.end(); ++mg_it )
    {
        delete *mg_it;    // each MergeGroup is deleted; its ptrs to PSimHits we do not own...
    }

    return product;
}


//------------------------------------------------------------------------------
//   Smear one hit.  The main action is in here.
//------------------------------------------------------------------------------
FastSingleTrackerRecHit PixelTemplateSmearerBase::smearHit(
    const PSimHit& simHit,
    const PixelGeomDetUnit* detUnit,
    const double boundX,
    const double boundY,
    RandomEngineAndDistribution const* random) const
{

    // at the beginning the position is the Local Point in the local pixel module reference frame
    // same code as in PixelCPEBase
    LocalVector localDir = simHit.momentumAtEntry().unit();
    float locx = localDir.x();
    float locy = localDir.y();
    float locz = localDir.z();

    float cotalpha = locx/locz;
    float cotbeta = locy/locz;
    float sign=1.;
    
    if(isForward)
    {
        if( cotbeta < 0 )
        {
            sign=-1.;
        }
        cotbeta = sign*cotbeta;
    }

    const PixelTopology* theSpecificTopology = &(detUnit->specificType().specificTopology());
    const RectangularPixelTopology *rectPixelTopology = static_cast<const RectangularPixelTopology*>(theSpecificTopology);

    const int nrows = theSpecificTopology->nrows();
    const int ncolumns = theSpecificTopology->ncolumns();

    const Local3DPoint lp = simHit.localPosition();
    //Transform local position to measurement position
    const MeasurementPoint mp = rectPixelTopology->measurementPosition( lp );
    float mpy = mp.y();
    float mpx = mp.x();
    //Get the center of the struck pixel in measurement position
    float pixelCenterY = 0.5 + (int)mpy;
    float pixelCenterX = 0.5 + (int)mpx;

    const MeasurementPoint mpCenter(pixelCenterX, pixelCenterY);
    //Transform the center of the struck pixel back into local position
    const Local3DPoint lpCenter = rectPixelTopology->localPosition( mpCenter );

    //Get the relative position of struck point to the center of the struck pixel
    float xtrk = lp.x() - lpCenter.x();
    float ytrk = lp.y() - lpCenter.y();
    //Pixel Y, X pitch
    const float ysize={0.015}, xsize={0.01};
    //Variables for SiPixelTemplate input, see SiPixelTemplate reco
    float yhit = 20. + 8.*(ytrk/ysize);
    float xhit = 20. + 8.*(xtrk/xsize);
    int   ybin = (int)yhit;
    int	xbin = (int)xhit;
    float yfrac= yhit - (float)ybin;
    float xfrac= xhit - (float)xbin;
    //Protect againt ybin, xbin being outside of range [0-39]
    if( ybin < 0 )    ybin = 0;
    if( ybin > 39 )   ybin = 39;
    if( xbin < 0 )    xbin = 0;
    if( xbin > 39 )   xbin = 39; 

    //Variables for SiPixelTemplate output
    //qBin -- normalized pixel charge deposition
    float qbin_frac[4];
    //Single pixel cluster projection possibility
    float ny1_frac, ny2_frac, nx1_frac, nx2_frac;
    bool singlex = false, singley = false;
    SiPixelTemplate templ(thePixelTemp_);
    templ.interpolate(templateId, cotalpha, cotbeta);
    templ.qbin_dist(templateId, cotalpha, cotbeta, qbin_frac, ny1_frac, ny2_frac, nx1_frac, nx2_frac );
    int  nqbin;

    double xsizeProbability = random->flatShoot();
    double ysizeProbability = random->flatShoot();
    bool hitbigx = rectPixelTopology->isItBigPixelInX( (int)mpx );
    bool hitbigy = rectPixelTopology->isItBigPixelInY( (int)mpy );

    if( hitbigx ) 
    if( xsizeProbability < nx2_frac )  singlex = true;
    else singlex = false;
    else
    if( xsizeProbability < nx1_frac )  singlex = true;
    else singlex = false;

    if( hitbigy )
    if( ysizeProbability < ny2_frac )  singley = true;
    else singley = false;
    else
    if( ysizeProbability < ny1_frac )  singley = true;
    else singley = false;



    // random multiplicity for alpha and beta
    double qbinProbability = random->flatShoot();
    for(int i = 0; i<4; ++i)
    {
        nqbin = i;
        if (qbinProbability < qbin_frac[i])
        {
            break;
        }
    }

    //Store interpolated pixel cluster profile
    //BYSIZE, BXSIZE, const definition from SiPixelTemplate
    float ytempl[41][BYSIZE] = {{0}}, xtempl[41][BXSIZE] = {{0}} ;
    templ.ytemp(0, 40, ytempl);
    templ.xtemp(0, 40, xtempl);

    std::vector<double> ytemp(BYSIZE);
    for( int i=0; i<BYSIZE; ++i)
    {
        ytemp[i]=(1.-yfrac)*ytempl[ybin][i]+yfrac*ytempl[ybin+1][i];
    }

    std::vector<double> xtemp(BXSIZE);
    for(int i=0; i<BXSIZE; ++i)
    {
        xtemp[i]=(1.-xfrac)*xtempl[xbin][i]+xfrac*xtempl[xbin+1][i];
    }

    //Pixel readout threshold
    const float qThreshold = templ.s50()*2.0;

    //Cut away pixels below readout threshold
    //For cluster lengths calculation
    int offsetX1=0, offsetX2=0, offsetY1=0, offsetY2=0;
    int firstY, lastY, firstX, lastX;
    for( firstY = 0; firstY < BYSIZE; ++firstY )
    {
        bool yCluster = ytemp[firstY] > qThreshold;
        if(yCluster)
        {
            offsetY1 = BHY -firstY;
            break;
        }
    }
    for(lastY = firstY; lastY < BYSIZE; ++lastY)
    {
        bool yCluster = ytemp[lastY] > qThreshold;
        if(!yCluster)
        {
            lastY = lastY - 1;
            offsetY2 = lastY - BHY;
            break;
        }
    }

    for( firstX = 0; firstX < BXSIZE; ++firstX )
    {
        bool xCluster = xtemp[firstX] > qThreshold;
        if(xCluster)
        {
            offsetX1 = BHX - firstX;
            break;
        }
    }
    for( lastX = firstX; lastX < BXSIZE; ++ lastX )
    {
        bool xCluster = xtemp[lastX] > qThreshold;
        if(!xCluster)
        {
            lastX = lastX - 1;
            offsetX2 = lastX - BHX;
            break;
        }
    }


    //--- Prepare to return results
    Local3DPoint thePosition;  
    double       thePositionX; 
    double       thePositionY; 
    double       thePositionZ; 
    LocalError   theError;     
    double       theErrorX;    
    double       theErrorY;    

    //------------------------------
    //  Check if the cluster is near an edge.  If it protrudes
    //  outside the edge of the sensor, the truncate it and it will
    //  get significantly messed up.
    //------------------------------
    bool edge, edgex, edgey;
    //  bool bigx, bigy;

    int firstPixelInX = (int)mpx - offsetX1 ;
    int firstPixelInY = (int)mpy - offsetY1 ;
    int lastPixelInX  = (int)mpx + offsetX2 ;
    int lastPixelInY  = (int)mpy + offsetY2 ;
    firstPixelInX = (firstPixelInX >= 0) ? firstPixelInX : 0 ;
    firstPixelInY = (firstPixelInY >= 0) ? firstPixelInY : 0 ;
    lastPixelInX  = (lastPixelInX < nrows ) ? lastPixelInX : nrows-1 ;
    lastPixelInY  = (lastPixelInY < ncolumns ) ? lastPixelInY : ncolumns-1;

    edgex = rectPixelTopology->isItEdgePixelInX( firstPixelInX ) || rectPixelTopology->isItEdgePixelInX( lastPixelInX );
    edgey = rectPixelTopology->isItEdgePixelInY( firstPixelInY ) || rectPixelTopology->isItEdgePixelInY( lastPixelInY );
    edge = edgex || edgey;

    //  bigx = rectPixelTopology->isItBigPixelInX( firstPixelInX ) || rectPixelTopology->isItBigPixelInX( lastPixelInX );
    //  bigy = rectPixelTopology->isItBigPixelInY( firstPixelInY ) || rectPixelTopology->isItBigPixelInY( lastPixelInY );
    bool hasBigPixelInX = rectPixelTopology->containsBigPixelInX( firstPixelInX, lastPixelInX );
    bool hasBigPixelInY = rectPixelTopology->containsBigPixelInY( firstPixelInY, lastPixelInY );

    //Variables for SiPixelTemplate pixel hit error output
    float sigmay, sigmax, sy1, sy2, sx1, sx2;  
    templ.temperrors(
        templateId, cotalpha, cotbeta, nqbin,          // inputs
        sigmay, sigmax, sy1, sy2, sx1, sx2        // outputs
    );

    if(edge)
    {
        if(edgex && !edgey)
        {
            theErrorX = 23.0*microntocm;
            theErrorY = 39.0*microntocm;
        }
        else if(!edgex && edgey)
        {
            theErrorX = 24.0*microntocm;
            theErrorY = 96.0*microntocm;
        }
        else
        {
            theErrorX = 31.0*microntocm;
            theErrorY = 90.0*microntocm;
        }
    }  
    else 
    {
        if (singlex)
        {
            if (hitbigx)
            {
                theErrorX = sx2*microntocm;
            }
            else
            {
                theErrorX = sx1*microntocm;
            }
        }
        else
        {
            theErrorX = sigmax*microntocm;
        }
        if (singley)
        {
            if (hitbigy)
            {
                theErrorY = sy2*microntocm;
            }
            else
            {
                theErrorY = sy1*microntocm;
            }
        }
        else 
        {
            theErrorY = sigmay*microntocm;
        }
    }


    //add misalignment error
    const TrackerGeomDet* misalignmentDetUnit = getMisalignedGeometry().idToDet(detUnit->geographicalId());
    const LocalError& alignmentError = misalignmentDetUnit->localAlignmentError();
    if (alignmentError.valid())
    {
        theError = LocalError( 
            theErrorX*theErrorX+alignmentError.xx(), 
            alignmentError.xy(), 
            theErrorY*theErrorY+alignmentError.yy()
        );
    }
    else
    {
        theError = LocalError( 
            theErrorX*theErrorX, 
            0.0,
            theErrorY*theErrorY
        );
    }

    // Local Error is 2D: (xx,xy,yy), square of sigma in first an third position 
    // as for resolution matrix
    // Generate position
    // get resolution histograms
    int cotalphaHistBin = (int)( ( cotalpha - rescotAlpha_binMin ) / rescotAlpha_binWidth + 1 );
    int cotbetaHistBin  = (int)( ( cotbeta  - rescotBeta_binMin )  / rescotBeta_binWidth + 1 );
    // protection against out-of-range (undeflows and overflows)
    if (cotalphaHistBin < 1) cotalphaHistBin = 1; 
    if (cotbetaHistBin  < 1) cotbetaHistBin  = 1; 
    if (cotalphaHistBin > (int)rescotAlpha_binN) cotalphaHistBin = (int)rescotAlpha_binN; 
    if (cotbetaHistBin  > (int)rescotBeta_binN) cotbetaHistBin  = (int)rescotBeta_binN; 
    //
    unsigned int theXHistN;
    unsigned int theYHistN;

    if (!isForward)
    {
        if (edge)
        {
            theXHistN = cotalphaHistBin * 1000 + cotbetaHistBin * 10	 +  (nqbin+1);
            theYHistN = theXHistN;	      
        }
        else
        {
            if (singlex)
            {
                if (hitbigx) theXHistN = 1 * 100000 + cotalphaHistBin * 100 + cotbetaHistBin ;
                else theXHistN = 1 * 10000 + cotbetaHistBin * 10 + cotalphaHistBin ; 
            }
            else
            {
                if (hasBigPixelInX) theXHistN = 1 * 1000000 + 1 * 100000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
                else theXHistN = 1 * 100000 + 1 * 10000 + cotbetaHistBin * 100 + cotalphaHistBin * 10 + (nqbin+1);
            }
            
            if(singley)
            {
                if (hitbigy) theYHistN = 1 * 100000 + cotalphaHistBin * 100 + cotbetaHistBin ;
                else theYHistN = 1 * 10000 + cotbetaHistBin * 10 + cotalphaHistBin ;
            }
            else
            {
                if (hasBigPixelInY) theYHistN = 1 * 1000000 + 1 * 100000 + cotalphaHistBin * 1000 + cotbetaHistBin * 10 + (nqbin+1);
                else theYHistN = 1 * 100000 + 1 * 10000 + cotbetaHistBin * 100 + cotalphaHistBin * 10 + (nqbin+1);
            }
        }
    }
    else
    {
        if (edge)
        {
            theXHistN = cotalphaHistBin * 1000 +  cotbetaHistBin * 10 +  (nqbin+1);
            theYHistN = theXHistN;
        }
        else
        {
            if (singlex)
            {
                if (hitbigx) theXHistN = 100000 + cotalphaHistBin * 100 + cotbetaHistBin;
                else theXHistN = cotbetaHistBin * 10 + cotalphaHistBin;
            }
            else
            {
                theXHistN = 10000 + cotbetaHistBin * 100 +  cotalphaHistBin * 10 +  (nqbin+1);    
            }
           
            if(singley)
            {
                if (hitbigy) theYHistN = 100000 + cotalphaHistBin * 100 + cotbetaHistBin;
                else theYHistN = cotbetaHistBin * 10 + cotalphaHistBin;
            }
            else
            {
                theYHistN = 10000 + cotbetaHistBin * 100 +  cotalphaHistBin * 10 + (nqbin+1);
            }
        }
    }
    
    
    unsigned int retry = 0;
    
    do 
    {
        //
        // Smear the hit Position

        std::map<unsigned int, const SimpleHistogramGenerator*>::const_iterator xgenIt = theXHistos.find(theXHistN);
        std::map<unsigned int, const SimpleHistogramGenerator*>::const_iterator ygenIt = theYHistos.find(theYHistN);
        if (xgenIt==theXHistos.cend() || ygenIt==theYHistos.cend())
        {
            throw cms::Exception("FastSimulation/TrackingRecHitProducer") << "Histogram ("<<theXHistN<<","<<theYHistN<<") was not found for PixelTemplateSmearer. Check if the smearing template exists.";
        }


        const SimpleHistogramGenerator* xgen = xgenIt->second;
        const SimpleHistogramGenerator* ygen = ygenIt->second;

        thePositionX = xgen->generate(random);
        thePositionY = ygen->generate(random);


        if( isForward )
        {
            thePositionY *= sign;
        }
        thePositionZ = 0.0; // set at the centre of the active area

        thePosition = Local3DPoint(
            simHit.localPosition().x() + thePositionX, 
            simHit.localPosition().y() + thePositionY, 
            simHit.localPosition().z() + thePositionZ
        );
        retry++;
        if (retry > 10) 
        {
            // If we tried to generate thePosition, and it's out of the bounds
            // for 10 times, then take and return the simHit's location.
            thePosition = Local3DPoint(
                simHit.localPosition().x(), 
                simHit.localPosition().y(), 
                simHit.localPosition().z()
            );
            break;
        }
    } while(fabs(thePosition.x()) > boundX  || fabs(thePosition.y()) > boundY);
    
    
    FastSingleTrackerRecHit recHit(
        thePosition,
        theError,
        *detUnit,
        fastTrackerRecHitType::siPixel
    );
    return recHit;
}


TrackingRecHitProductPtr PixelTemplateSmearerBase::
processUnmergedHits(
    std::vector<TrackingRecHitProduct::SimHitIdPair> & unmergedHits, 
    TrackingRecHitProductPtr product,
    const PixelGeomDetUnit * detUnit,
    const double boundX, const double boundY,
    RandomEngineAndDistribution const * random
) const
{
    for (auto simHitIdPair: unmergedHits)
    {
        FastSingleTrackerRecHit recHit = smearHit( *simHitIdPair.second, detUnit, boundX, boundY, random );
        product->addRecHit( recHit ,{simHitIdPair});
    }
    return product;
}


TrackingRecHitProductPtr PixelTemplateSmearerBase::
processMergeGroups(
    std::vector< MergeGroup* > & mergeGroups,
    TrackingRecHitProductPtr product,
    const PixelGeomDetUnit * detUnit,
    const double boundX, const double boundY,
    RandomEngineAndDistribution const * random
) const
{
    for ( auto mg_it = mergeGroups.begin(); mg_it != mergeGroups.end(); ++mg_it) 
    {
        if ((*mg_it)->smearIt)
        {
            FastSingleTrackerRecHit recHit = smearMergeGroup( *mg_it, detUnit, boundX, boundY, random );
            product->addRecHit( recHit, (*mg_it)->group);
        }
    }
    return product;
}


FastSingleTrackerRecHit PixelTemplateSmearerBase::
smearMergeGroup(
    MergeGroup* mg,
    const PixelGeomDetUnit * detUnit,
    const double boundX, const double boundY,
    RandomEngineAndDistribution const * random
) const 
{

    float loccx = 0;
    float loccy = 0;
    float loccz = 0;
    float nHit = 0;
    float locpx = 0;
    float locpy = 0;
    float locpz = 0;

    for (auto hit_it = mg->group.begin(); hit_it != mg->group.end(); ++hit_it) 
    {
        const PSimHit simHit = *hit_it->second; 
        //getting local momentum and adding all of the hits' momentums up
        LocalVector localDir = simHit.momentumAtEntry().unit();
        loccx += localDir.x();
        loccy += localDir.y();
        loccz += localDir.z();
        //getting local position and adding all of the hits' positions up
        const Local3DPoint lpos = simHit.localPosition();
        locpx += lpos.x();
        locpy += lpos.y();
        locpz += lpos.z();
        //counting how many sim hits are in the merge group
        nHit += 1;
    }
    //averaging the momentums by diving momentums added up/number of hits
    float locx = loccx/nHit;
    float locy = loccy/nHit;
    float locz = loccz/nHit;

    // alpha: angle with respect to local x axis in local (x,z) plane
    float cotalpha = locx/locz;
    // beta: angle with respect to local y axis in local (y,z) plane
    float cotbeta = locy/locz;
    float sign=1.;

    if( isForward )
    {
        if( cotbeta < 0 )
        {
            sign=-1.;
        }
        cotbeta = sign*cotbeta;
    }


    float lpx = locpx/nHit;
    float lpy = locpy/nHit;
    float lpz = locpz/nHit;

    //Get the relative position of struck point to the center of the struck pixel
    float xtrk = lpx;
    float ytrk = lpy;
    //Pixel Y, X pitch
    const float ysize={0.015}, xsize={0.01};
    //Variables for SiPixelTemplate input, see SiPixelTemplate reco
    float yhit = 20. + 8.*(ytrk/ysize);
    float xhit = 20. + 8.*(xtrk/xsize);
    int   ybin = (int)yhit;
    int	xbin = (int)xhit;
    float yfrac= yhit - (float)ybin;
    float xfrac= xhit - (float)xbin;
    //Protect againt ybin, xbin being outside of range [0-39]
    if( ybin < 0 )    ybin = 0;
    if( ybin > 39 )   ybin = 39;
    if( xbin < 0 )    xbin = 0;
    if( xbin > 39 )   xbin = 39; 

    //Variables for SiPixelTemplate output
    //qBin -- normalized pixel charge deposition
    float qbin_frac[4];
    //Single pixel cluster projection possibility
    float ny1_frac, ny2_frac, nx1_frac, nx2_frac;
    bool singlex = false, singley = false;
    SiPixelTemplate templ(thePixelTemp_);
    templ.interpolate(templateId, cotalpha, cotbeta);
    templ.qbin_dist(templateId, cotalpha, cotbeta, qbin_frac, ny1_frac, ny2_frac, nx1_frac, nx2_frac );
    int  nqbin;

    //  double xsizeProbability = random->flatShoot();
    //double ysizeProbability = random->flatShoot();
    bool hitbigx = false;
    bool hitbigy = false;



    // random multiplicity for alpha and beta
    double qbinProbability = random->flatShoot();
    for(int i = 0; i<4; ++i)
    {
        nqbin = i;
        if(qbinProbability < qbin_frac[i]) break;
    }

    //Store interpolated pixel cluster profile
    //BYSIZE, BXSIZE, const definition from SiPixelTemplate
    float ytempl[41][BYSIZE] = {{0}}, xtempl[41][BXSIZE] = {{0}} ;
    templ.ytemp(0, 40, ytempl);
    templ.xtemp(0, 40, xtempl);

    std::vector<double> ytemp(BYSIZE);
    for( int i=0; i<BYSIZE; ++i)
    {
        ytemp[i]=(1.-yfrac)*ytempl[ybin][i]+yfrac*ytempl[ybin+1][i];
    }

    std::vector<double> xtemp(BXSIZE);
    for(int i=0; i<BXSIZE; ++i)
    {
        xtemp[i]=(1.-xfrac)*xtempl[xbin][i]+xfrac*xtempl[xbin+1][i];
    }

    //--- Prepare to return results
    Local3DPoint thePosition;  
    double       thePositionX; 
    double       thePositionY; 
    double       thePositionZ; 
    LocalError   theError;     
    double       theErrorX;    
    double       theErrorY;    
    //double       theErrorZ;    



    //------------------------------
    //  Check if the cluster is near an edge.  If it protrudes
    //  outside the edge of the sensor, the truncate it and it will
    //  get significantly messed up.
    //------------------------------
    bool edge = false;
    bool edgex = false;
    bool edgey = false;


    //Variables for SiPixelTemplate pixel hit error output
    float sigmay, sigmax, sy1, sy2, sx1, sx2;  
    templ.temperrors(templateId, cotalpha, cotbeta, nqbin,          // inputs
	       sigmay, sigmax, sy1, sy2, sx1, sx2 );      // outputs

    // define private mebers --> Errors
    if (edge)
    {
        if (edgex && !edgey)
        {
            theErrorX = 23.0*microntocm;
            theErrorY = 39.0*microntocm;
        }
        else if( !edgex && edgey )
        {
            theErrorX = 24.0*microntocm;
            theErrorY = 96.0*microntocm;
        }
        else
        {
            theErrorX = 31.0*microntocm;
            theErrorY = 90.0*microntocm;
        }

    }  
    else
    {
        if( singlex )
        {
            if ( hitbigx )
            {
                theErrorX = sx2*microntocm;
            }
            else
            {
                theErrorX = sx1*microntocm;
            }
        }
        else  
        {
            theErrorX = sigmax*microntocm;
        }
        
        if (singley)
        {
            if (hitbigy)
            {
                theErrorY = sy2*microntocm;
            }
            else
            {
                theErrorY = sy1*microntocm;
            }
        }
        else
        {
            theErrorY = sigmay*microntocm;
        }
    }
    
    theError = LocalError( theErrorX*theErrorX, 0., theErrorY*theErrorY);

    
    unsigned int retry = 0;
    do
    {
        const SimpleHistogramGenerator* xgen = new SimpleHistogramGenerator( (TH1F*) theMergedPixelResolutionXFile-> Get("th1x")); 
        const SimpleHistogramGenerator* ygen = new SimpleHistogramGenerator( (TH1F*) theMergedPixelResolutionYFile-> Get("th1y")); 

        thePositionX = xgen->generate(random);
        thePositionY = ygen->generate(random);

        if( isForward )
        {
            thePositionY *= sign;
        }
        thePositionZ = 0.0; // set at the centre of the active area
        thePosition = 
	           Local3DPoint(lpx + thePositionX , 
                       lpy + thePositionY , 
                       lpz + thePositionZ );
        
        retry++;
        if (retry > 10)
        {
            // If we tried to generate thePosition, and it's out of the bounds
            // for 10 times, then take and return the simHit's location.
            thePosition = Local3DPoint(lpx,lpy,lpz);
            break;
        }
    } while(fabs(thePosition.x()) > boundX  || fabs(thePosition.y()) > boundY);

    FastSingleTrackerRecHit recHit(
        thePosition,
        theError,
        *detUnit,
        fastTrackerRecHitType::siPixel
    );
    return recHit;
}


bool PixelTemplateSmearerBase::hitsMerge(const PSimHit& simHit1,const PSimHit& simHit2) const
{
    LocalVector localDir = simHit1.momentumAtEntry().unit();
    float locy1 = localDir.y();
    float locz1 = localDir.z();
    float cotbeta = locy1/locz1;
    float loceta = fabs(-log((double)(-cotbeta+sqrt((double)(1.+cotbeta*cotbeta)))));

    const Local3DPoint lp1 = simHit1.localPosition();
    const Local3DPoint lp2 = simHit2.localPosition();
    float lpy1 = lp1.y();
    float lpx1 = lp1.x();
    float lpy2 = lp2.y();
    float lpx2 = lp2.x();
    float locdis = 10000.* sqrt(pow(lpx1 - lpx2, 2) + pow(lpy1 - lpy2, 2));
    TH2F * probhisto = (TH2F*)theMergingProbabilityFile->Get("h2bc");
    float prob = probhisto->GetBinContent(probhisto->GetXaxis()->FindFixBin(locdis),probhisto->GetYaxis()->FindFixBin(loceta));
    return prob > 0;
}


//-----------------------------------------------------------------------------
// The isFlipped() is a silly way to determine which detectors are inverted.
// In the barrel for every 2nd ladder the E field direction is in the
// global r direction (points outside from the z axis), every other
// ladder has the E field inside. Something similar is in the 
// forward disks (2 sides of the blade). This has to be recognised
// because the charge sharing effect is different.
//
// The isFliped does it by looking and the relation of the local (z always
// in the E direction) to global coordinates. There is probably a much 
// better way.(PJ: And faster!)
//-----------------------------------------------------------------------------
bool PixelTemplateSmearerBase::isFlipped(const PixelGeomDetUnit* theDet) const
{
    float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
    float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
    return tmp2<tmp1;
}


