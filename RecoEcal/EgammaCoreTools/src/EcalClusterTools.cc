#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "CLHEP/Geometry/Transform3D.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/EcalBarrelAlgo/interface/EcalBarrelGeometry.h"

std::pair<DetId, float> EcalClusterTools::getMaximum( const std::vector<DetId> &v_id, const EcalRecHitCollection *recHits)
{
        float max = 0;
        DetId id(0);
        for ( size_t i = 0; i < v_id.size(); ++i ) {
                float energy = recHitEnergy( v_id[i], recHits );
                if ( energy > max ) {
                        max = energy;
                        id = v_id[i];
                }
        }
        return std::pair<DetId, float>(id, max);
}



std::pair<DetId, float> EcalClusterTools::getMaximum( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits)
{
        return getMaximum( cluster.getHitsByDetId(), recHits );
}



float EcalClusterTools::recHitEnergy(DetId id, const EcalRecHitCollection *recHits)
{
        if ( id == DetId(0) ) {
                return 0;
        } else {
                EcalRecHitCollection::const_iterator it = recHits->find( id );
                if ( it != recHits->end() ) {
                        return (*it).energy();
                } else {
                        // FIXME : throw exception !
                }
        }
        return 0;
}



float EcalClusterTools::matrixEnergy( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
        // fast version
        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology );
        float energy = 0;
        for ( int i = ixMin; i <= ixMax; ++i ) {
                for ( int j = iyMin; j <= iyMax; ++j ) {
                        cursor.home();
                        cursor.offsetBy( i, j );
                        energy += recHitEnergy( *cursor, recHits );
                }
        }
        // slow elegant version
        //float energy = 0;
        //std::vector<DetId> v_id = matrixDetId( topology, id, ixMin, ixMax, iyMin, iyMax );
        //for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {
        //        energy += recHitEnergy( *it, recHits );
        //}
        return energy;
}



std::vector<DetId> EcalClusterTools::matrixDetId( const CaloSubdetectorTopology* topology, DetId id, int ixMin, int ixMax, int iyMin, int iyMax )
{
        CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology );
        std::vector<DetId> v;
        for ( int i = ixMin; i <= ixMax; ++i ) {
                for ( int j = iyMin; j <= iyMax; ++j ) {
                        cursor.home();
                        cursor.offsetBy( i, j );
                        v.push_back( *cursor );
                }
        }
        return v;
}



float EcalClusterTools::e2x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        std::list<float> energies;
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 0 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0,  0, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1,  0, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 0 ) );
        energies.sort();
        return *--energies.end();
}



float EcalClusterTools::e3x2( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        std::list<float> energies;
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 0 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id,  0, 1, -1, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 1,  0, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 0, -1, 1 ) );
        energies.sort();
        return *--energies.end();
}



float EcalClusterTools::e3x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -1, 1, -1, 1 );
}



float EcalClusterTools::e4x4( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        std::list<float> energies;
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -2, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -2, 1 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -2, 1, -1, 2 ) );
        energies.push_back( matrixEnergy( cluster, recHits, topology, id, -1, 2, -1, 2 ) );
        energies.sort();
        return *--energies.end();
}



float EcalClusterTools::e5x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -2, 2, -2, 2 );
}



float EcalClusterTools::eMax( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
        return getMaximum( cluster.getHitsByDetId(), recHits ).second;
}



float EcalClusterTools::e2nd( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
        std::list<float> energies;
        std::vector<DetId> v_id = cluster.getHitsByDetId();
        if ( v_id.size() < 2 ) return 0;
        for ( size_t i = 0; i < v_id.size(); ++i ) {
                energies.push_back( recHitEnergy( v_id[i], recHits ) );
        }
        energies.sort();
        return *--(--energies.end());
}



float EcalClusterTools::e2x5Right( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, 1, 2, -2, 2 );
}



float EcalClusterTools::e2x5Left( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -2, -1, -2, 2 );
}



float EcalClusterTools::e2x5Top( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -2, 2, 1, 2 );
}



float EcalClusterTools::e2x5Bottom( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -2, -2, -2, -1 );
}



float EcalClusterTools::e1x5( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -2, -2, 0, 0 );
}



float EcalClusterTools::e5x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, 0, 0, -2, 2 );
}



float EcalClusterTools::e1x3( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, -1, 1, 0, 0 );
}



float EcalClusterTools::e3x1( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology* topology )
{
        DetId id = getMaximum( cluster.getHitsByDetId(), recHits ).first;
        return matrixEnergy( cluster, recHits, topology, id, 0, 0, -1, 1 );
}



std::vector<float> EcalClusterTools::energyBasketFractionEta( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
        std::vector<float> basketFraction( 2 * EBDetId::kModulesPerSM );
        float clusterEnergy = cluster.energy();
        std::vector<DetId> v_id = cluster.getHitsByDetId();
        if ( v_id[0].subdetId() != EcalBarrel ) {
                // FIXME: throw exception
        }
        for ( size_t i = 0; i < v_id.size(); ++i ) {
                basketFraction[ EBDetId(v_id[i]).im()-1 + EBDetId(v_id[i]).positiveZ()*EBDetId::kModulesPerSM ] = recHitEnergy( v_id[i], recHits ) / clusterEnergy;
        }
        std::sort( basketFraction.rbegin(), basketFraction.rend() );
        return basketFraction;
}



std::vector<float> EcalClusterTools::energyBasketFractionPhi( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits )
{
        std::vector<float> basketFraction( 2 * (EBDetId::MAX_IPHI / EBDetId::kCrystalsInPhi) );
        float clusterEnergy = cluster.energy();
        std::vector<DetId> v_id = cluster.getHitsByDetId();
        if ( v_id[0].subdetId() != EcalBarrel ) {
                // FIXME: throw exception
        }
        for ( size_t i = 0; i < v_id.size(); ++i ) {
                basketFraction[ (EBDetId(v_id[i]).iphi()-1)/EBDetId::kCrystalsInPhi + EBDetId(v_id[i]).positiveZ()*EBDetId::kTowersInPhi] = recHitEnergy( v_id[i], recHits ) / clusterEnergy;
        }
        std::sort( basketFraction.rbegin(), basketFraction.rend() );
        return basketFraction;
}



std::vector<EcalClusterTools::EcalClusterEnergyDeposition> EcalClusterTools::getEnergyDepTopology( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, bool logW, float w0 )
{
        std::vector<EcalClusterTools::EcalClusterEnergyDeposition> energyDistribution;
        // init a map of the energy deposition centered on the
        // cluster centroid. This is for momenta calculation only.
        CLHEP::Hep3Vector clVect(cluster.position().x(), cluster.position().y(), cluster.position().z());
        CLHEP::Hep3Vector clDir(clVect);
        clDir*=1.0/clDir.mag();
        // in the transverse plane, axis perpendicular to clusterDir
        CLHEP::Hep3Vector theta_axis(clDir.y(),-clDir.x(),0.0);
        theta_axis *= 1.0/theta_axis.mag();
        Hep3Vector phi_axis = theta_axis.cross(clDir);

        std::vector<DetId> clusterDetIds = cluster.getHitsByDetId();

        EcalClusterEnergyDeposition clEdep;
        EcalRecHit testEcalRecHit;
        std::vector<DetId>::iterator posCurrent;
        // loop over crystals
        for(posCurrent=clusterDetIds.begin(); posCurrent!=clusterDetIds.end(); ++posCurrent) {
                EcalRecHitCollection::const_iterator itt = recHits->find(*posCurrent);
                testEcalRecHit=*itt;

                if((*posCurrent != DetId(0)) && (recHits->find(*posCurrent) != recHits->end())) {
                        clEdep.deposited_energy = testEcalRecHit.energy();
                        // if logarithmic weight is requested, apply cut on minimum energy of the recHit
                        if(logW) {
                                //double w0 = parameterMap_.find("W0")->second;

                                double weight = std::max(0.0, w0 + log(fabs(clEdep.deposited_energy)/cluster.energy()) );
                                if(weight==0) {
                                        LogDebug("ClusterShapeAlgo") << "Crystal has insufficient energy: E = " 
                                                << clEdep.deposited_energy << " GeV; skipping... ";
                                        continue;
                                }
                                else LogDebug("ClusterShapeAlgo") << "===> got crystal. Energy = " << clEdep.deposited_energy << " GeV. ";
                        }
                        DetId id_ = *posCurrent;
                        const CaloCellGeometry *this_cell = geometry->getGeometry(id_);
                        GlobalPoint cellPos = this_cell->getPosition();
                        CLHEP::Hep3Vector gblPos (cellPos.x(),cellPos.y(),cellPos.z()); //surface position?
                        // Evaluate the distance from the cluster centroid
                        CLHEP::Hep3Vector diff = gblPos - clVect;
                        // Important: for the moment calculation, only the "lateral distance" is important
                        // "lateral distance" r_i = distance of the digi position from the axis Origin-Cluster Center
                        // ---> subtract the projection on clDir
                        CLHEP::Hep3Vector DigiVect = diff - diff.dot(clDir)*clDir;
                        clEdep.r = DigiVect.mag();
                        LogDebug("ClusterShapeAlgo") << "E = " << clEdep.deposited_energy
                                << "\tdiff = " << diff.mag()
                                << "\tr = " << clEdep.r;
                        clEdep.phi = DigiVect.angle(theta_axis);
                        if(DigiVect.dot(phi_axis)<0) clEdep.phi = 2 * M_PI - clEdep.phi;
                        energyDistribution.push_back(clEdep);
                }
        } 
        return energyDistribution;
}



std::pair<float, float> EcalClusterTools::etaPhiLat( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, bool logW, float w0 )
{
        std::vector<EcalClusterTools::EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );

        double r, redmoment=0;
        double phiRedmoment = 0 ;
        double etaRedmoment = 0 ;
        int n,n1,n2,tmp;
        int clusterSize=energyDistribution.size();
        float etaLat_, phiLat_, lat_;
        if (clusterSize<3) {
                etaLat_ = 0.0 ; 
                lat_ = 0.0;
                return std::pair<float, float>(0., 0.); 
        }

        n1=0; n2=1;
        if (energyDistribution[1].deposited_energy > 
                        energyDistribution[0].deposited_energy) 
        {
                tmp=n2; n2=n1; n1=tmp;
        }
        for (int i=2; i<clusterSize; i++) {
                n=i;
                if (energyDistribution[i].deposited_energy > 
                                energyDistribution[n1].deposited_energy) 
                {
                        tmp = n2;
                        n2 = n1; n1 = i; n=tmp;
                } else {
                        if (energyDistribution[i].deposited_energy > 
                                        energyDistribution[n2].deposited_energy) 
                        {
                                tmp=n2; n2=i; n=tmp;
                        }
                }

                r = energyDistribution[n].r;
                redmoment += r*r* energyDistribution[n].deposited_energy;
                double rphi = r * cos (energyDistribution[n].phi) ;
                phiRedmoment += rphi * rphi * energyDistribution[n].deposited_energy;
                double reta = r * sin (energyDistribution[n].phi) ;
                etaRedmoment += reta * reta * energyDistribution[n].deposited_energy;
        } 
        double e1 = energyDistribution[n1].deposited_energy;
        double e2 = energyDistribution[n2].deposited_energy;

        lat_ = redmoment/(redmoment+2.19*2.19*(e1+e2));
        phiLat_ = phiRedmoment/(phiRedmoment+2.19*2.19*(e1+e2));
        etaLat_ = etaRedmoment/(etaRedmoment+2.19*2.19*(e1+e2));

        return std::pair<float, float>( phiLat_, etaLat_);
}



math::XYZVector EcalClusterTools::meanClusterPosition( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorTopology *topology, const CaloSubdetectorGeometry *geometry )
{
        // find mean energy position of a 5x5 cluster around the maximum
        math::XYZVector meanPosition(0.0, 0.0, 0.0);
        std::vector<DetId> v_id = matrixDetId( topology, getMaximum( cluster, recHits ).first, -2, 2, -2, 2 );
        for ( std::vector<DetId>::const_iterator it = v_id.begin(); it != v_id.end(); ++it ) {
                GlobalPoint positionGP = geometry->getGeometry( *it )->getPosition();
                math::XYZVector position(positionGP.x(),positionGP.y(),positionGP.z());
                meanPosition = meanPosition + recHitEnergy( *it, recHits ) * position;
        }
        return meanPosition / e5x5( cluster, recHits, topology );
}



std::vector<float> EcalClusterTools::covariances(const reco::BasicCluster &cluster, const EcalRecHitCollection* recHits, const CaloSubdetectorTopology *topology, const CaloSubdetectorGeometry* geometry, float w0)
{
        float e_5x5 = e5x5( cluster, recHits, topology );
        float covEtaEta, covEtaPhi, covPhiPhi;
        if (e_5x5 > 0.) {
                //double w0_ = parameterMap_.find("W0")->second;
                std::vector<DetId> v_id = cluster.getHitsByDetId();
                math::XYZVector meanPosition = meanClusterPosition( cluster, recHits, topology, geometry );

                // now we can calculate the covariances
                double numeratorEtaEta = 0;
                double numeratorEtaPhi = 0;
                double numeratorPhiPhi = 0;
                double denominator     = 0;

                CaloNavigator<DetId> cursor = CaloNavigator<DetId>( getMaximum( v_id, recHits ).first, topology );
                for ( int i = -2; i <= 2; ++i ) {
                        for ( int j = -2; j <= 2; ++j ) {
                                cursor.home();
                                cursor.offsetBy( i, j );
                                float energy = recHitEnergy( *cursor, recHits );

                                if ( energy <= 0 ) continue;

                                GlobalPoint position = geometry->getGeometry(*cursor)->getPosition();

                                double dPhi = position.phi() - meanPosition.phi();
                                if (dPhi > + Geom::pi()) { dPhi = Geom::twoPi() - dPhi; }
                                if (dPhi < - Geom::pi()) { dPhi = Geom::twoPi() + dPhi; }

                                double dEta = position.eta() - meanPosition.eta();
                                double w = 0.;
                                w = std::max(0.0, w0 + log( energy / e_5x5 ));

                                denominator += w;
                                numeratorEtaEta += w * dEta * dEta;
                                numeratorEtaPhi += w * dEta * dPhi;
                                numeratorPhiPhi += w * dPhi * dPhi;
                        }
                }

                covEtaEta = numeratorEtaEta / denominator;
                covEtaPhi = numeratorEtaPhi / denominator;
                covPhiPhi = numeratorPhiPhi / denominator;
        } else {
                // Warn the user if there was no energy in the cells and return zeroes.
                //       std::cout << "\ClusterShapeAlgo::Calculate_Covariances:  no energy in supplied cells.\n";
                covEtaEta = 0;
                covEtaPhi = 0;
                covPhiPhi = 0;
        }
        std::vector<float> v;
        v.push_back( covEtaEta );
        v.push_back( covEtaPhi );
        v.push_back( covPhiPhi );
        return v;
}



double EcalClusterTools::zernike20( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, double R0, bool logW, float w0 )
{
        return absZernikeMoment( cluster, recHits, geometry, 2, 0, R0, logW, w0 );
}



double EcalClusterTools::zernike42( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, double R0, bool logW, float w0 )
{
        return absZernikeMoment( cluster, recHits, geometry, 4, 2, R0, logW, w0 );
}



double EcalClusterTools::absZernikeMoment( const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
        // 1. Check if n,m are correctly
        if ((m>n) || ((n-m)%2 != 0) || (n<0) || (m<0)) return -1;

        // 2. Check if n,R0 are within validity Range :
        // n>20 or R0<2.19cm  just makes no sense !
        if ((n>20) || (R0<=2.19)) return -1;
        if (n<=5) return fast_AbsZernikeMoment(cluster, recHits, geometry, n, m, R0, logW, w0 );
        else return calc_AbsZernikeMoment(cluster, recHits, geometry, n, m, R0, logW, w0 );
}


double EcalClusterTools::fast_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
        double r,ph,e,Re=0,Im=0;
        double TotalEnergy = cluster.energy();
        int index = (n/2)*(n/2)+(n/2)+m;
        std::vector<EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );
        int clusterSize = energyDistribution.size();
        if(clusterSize < 3) return 0.0;

        for (int i=0; i<clusterSize; i++)
        { 
                r = energyDistribution[i].r / R0;
                if (r<1) {
                        std::vector<double> pol;
                        pol.push_back( f00(r) );
                        pol.push_back( f11(r) );
                        pol.push_back( f20(r) );
                        pol.push_back( f22(r) );
                        pol.push_back( f31(r) );
                        pol.push_back( f33(r) );
                        pol.push_back( f40(r) );
                        pol.push_back( f42(r) );
                        pol.push_back( f44(r) );
                        pol.push_back( f51(r) );
                        pol.push_back( f53(r) );
                        pol.push_back( f55(r) );
                        ph = (energyDistribution[i]).phi;
                        e = energyDistribution[i].deposited_energy;
                        Re = Re + e/TotalEnergy * pol[index] * cos( (double) m * ph);
                        Im = Im - e/TotalEnergy * pol[index] * sin( (double) m * ph);
                }
        }
        return sqrt(Re*Re+Im*Im);
}



double EcalClusterTools::calc_AbsZernikeMoment(const reco::BasicCluster &cluster, const EcalRecHitCollection *recHits, const CaloSubdetectorGeometry *geometry, int n, int m, double R0, bool logW, float w0 )
{
        double r, ph, e, Re=0, Im=0, f_nm;
        double TotalEnergy = cluster.energy();
        std::vector<DetId> clusterDetIds = cluster.getHitsByDetId();
        std::vector<EcalClusterEnergyDeposition> energyDistribution = getEnergyDepTopology( cluster, recHits, geometry, logW, w0 );
        int clusterSize=energyDistribution.size();
        if(clusterSize<3) return 0.0;

        for (int i = 0; i < clusterSize; ++i)
        { 
                r = energyDistribution[i].r / R0;
                if (r < 1) {
                        ph = energyDistribution[i].phi;
                        e = energyDistribution[i].deposited_energy;
                        f_nm = 0;
                        for (int s=0; s<=(n-m)/2; s++) {
                                if (s%2==0) { 
                                        f_nm = f_nm + factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
                                } else {
                                        f_nm = f_nm - factorial(n-s)*pow(r,(double) (n-2*s))/(factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
                                }
                        }
                        Re = Re + e/TotalEnergy * f_nm * cos( (double) m*ph);
                        Im = Im - e/TotalEnergy * f_nm * sin( (double) m*ph);
                }
        }
        return sqrt(Re*Re+Im*Im);
}
