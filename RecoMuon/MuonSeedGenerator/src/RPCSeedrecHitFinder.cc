/**
 *  See header file for a description of this class.
 *
 */


#include "RecoMuon/MuonSeedGenerator/src/RPCSeedrecHitFinder.h"
#include <DataFormats/TrackingRecHit/interface/TrackingRecHit.h>
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>

using namespace std;
using namespace edm;


// Comparator function must be global function?
// Could not be included in .h file, or there will be 2 lessPhi() functions in both RPCSeedGenerator and RPCSeedFinder module
bool lessPhi(const MuonTransientTrackingRecHit::ConstMuonRecHitPointer& it1, const MuonTransientTrackingRecHit::ConstMuonRecHitPointer& it2)
{
    // Don't need to use value() in Geom::Phi to short
    return (it1->globalPosition().phi() < it2->globalPosition().phi());
}

RPCSeedrecHitFinder::RPCSeedrecHitFinder() {

    // Initiate the member
    isLayerset = false;
    isConfigured = false;
    isInputset = false;
    isOutputset = false;
    BxRange = 0;
    MaxDeltaPhi = 0;
    ClusterSet.clear();
    LayersinRPC.clear();
    therecHits.clear();
}

RPCSeedrecHitFinder::~RPCSeedrecHitFinder() {

}

void RPCSeedrecHitFinder::configure(const edm::ParameterSet& iConfig) {

    // Set the configuration
    BxRange = iConfig.getParameter<unsigned int>("BxRange");
    MaxDeltaPhi = iConfig.getParameter<double>("MaxDeltaPhi");
    ClusterSet = iConfig.getParameter< std::vector<int> >("ClusterSet");

    // Set the signal open
    isConfigured = true;
}

void RPCSeedrecHitFinder::setInput(MuonRecHitContainer (&recHits)[RPCLayerNumber]) {

    for(unsigned int i = 0; i < RPCLayerNumber; i++)
        recHitsRPC[i] = &recHits[i];
    isInputset = true;
}

void RPCSeedrecHitFinder::unsetInput() {

    isInputset = false;
}
void RPCSeedrecHitFinder::setOutput(RPCSeedFinder *Seed) {

    theSeed = Seed;
    isOutputset = true;
}

void RPCSeedrecHitFinder::setLayers(const std::vector<unsigned int>& Layers) {

    LayersinRPC = Layers;
    isLayerset = true;
}

void RPCSeedrecHitFinder::fillrecHits() {

    if(isLayerset == false || isConfigured == false || isOutputset == false || isInputset == false)
    {
        cout << "Not set the IO or not configured yet" << endl;
        return;
    }
    cout << "Now fill recHits from Layers: ";
    for(unsigned int k = 0; k < LayersinRPC.size(); k++)
        cout << LayersinRPC[k] <<" ";
    cout << endl;
    unsigned int LayerIndex = 0;
    therecHits.clear();
    complete(LayerIndex);

    // Unset the signal
    LayersinRPC.clear();
    therecHits.clear();
    isLayerset = false;
}

void RPCSeedrecHitFinder::complete(unsigned int LayerIndex) {

    for(MuonRecHitContainer::const_iterator it = recHitsRPC[LayersinRPC[LayerIndex]]->begin(); it != recHitsRPC[LayersinRPC[LayerIndex]]->end(); it++)  
    {
        cout << "Completing layer[" << LayersinRPC[LayerIndex] << "]." << endl;

        // Check validation
        if(!(*it)->isValid())
            continue;

        // Check BX range, be sure there is only RPCRecHit in the MuonRecHitContainer when use the dynamic_cast
        TrackingRecHit* thisTrackingRecHit = (*it)->hit()->clone();
        // Should also delete the RPCRecHit object cast by dynamic_cast<> ?
        RPCRecHit* thisRPCRecHit = dynamic_cast<RPCRecHit*>(thisTrackingRecHit);
        int BX = thisRPCRecHit->BunchX();
        int ClusterSize = thisRPCRecHit->clusterSize();
        delete thisTrackingRecHit;
        // Check BX
        if((unsigned int)abs(BX) > BxRange)
            continue;
        // Check cluster size
        bool Clustercheck = false;
        if(ClusterSet.size() == 0)
            Clustercheck = true;
        for(std::vector<int>::const_iterator CluIter = ClusterSet.begin(); CluIter != ClusterSet.end(); CluIter++)
            if(ClusterSize == (*CluIter))
                Clustercheck = true;
        if(Clustercheck != true)
            continue;
        // Check the recHits Phi range
        GlobalPoint pos = (*it)->globalPosition();
        double Phi = pos.phi();
        cout << "Phi: " << Phi << endl;
        // The recHits should locate in some phi range
        therecHits.push_back(*it);
        double deltaPhi = getdeltaPhifromrecHits();
        cout << "Delta phi: "<< deltaPhi << endl;
        therecHits.pop_back();
        if(deltaPhi > MaxDeltaPhi)
            continue;

        // If pass all, add to the seed
        therecHits.push_back(*it);
        cout << "RecHit's global position: " << pos.x() << ", " << pos.y() << ", " << pos.z() << endl;

        // Check if this recHit is the last one in the seed
        // If it is the last one, calculate the seed
        if(LayerIndex == (LayersinRPC.size()-1))
        {
            cout << "Check and fill one seed." << endl;
            checkandfill();
        }
        // If it is not the last one, continue to fill the seed from other layers
        else
            complete(LayerIndex+1);

        // Remember to pop the recHit before add another one from the same layer!
        therecHits.pop_back();
    }
}

double RPCSeedrecHitFinder::getdeltaPhifromrecHits() {

    ConstMuonRecHitContainer sortRecHits = therecHits;
    sort(sortRecHits.begin(), sortRecHits.end(), lessPhi);
    cout << "Sorted recHit's Phi: ";
    for(ConstMuonRecHitContainer::const_iterator iter = sortRecHits.begin(); iter != sortRecHits.end(); iter++)
        cout << (*iter)->globalPosition().phi() << ", ";
    cout << endl;
    // Calculate the deltaPhi, take care Geom::Phi always in range [-pi,pi)
    // In case of some deltaPhi larger then Pi, use value() in Geom::Phi to get the true value in radians of Phi, then do the calculation
    double deltaPhi = 0;
    if(sortRecHits.size() <= 1)
        return deltaPhi;
    if(sortRecHits.size() == 2)
    {
        ConstMuonRecHitContainer::const_iterator iter1 = sortRecHits.begin();
        ConstMuonRecHitContainer::const_iterator iter2 = sortRecHits.begin();
        iter2++;
        deltaPhi = (((*iter2)->globalPosition().phi().value() - (*iter1)->globalPosition().phi().value()) > M_PI) ? (2 * M_PI - ((*iter2)->globalPosition().phi().value() - (*iter1)->globalPosition().phi().value())) : ((*iter2)->globalPosition().phi().value() - (*iter1)->globalPosition().phi().value());
        return deltaPhi;
    }
    else
    {
        deltaPhi = 2 * M_PI;
        int n = 0;
        for(ConstMuonRecHitContainer::const_iterator iter = sortRecHits.begin(); iter != sortRecHits.end(); iter++)
        {   
            cout << "Before this loop deltaPhi is " << deltaPhi << endl;
            n++;
            double deltaPhi_more = 0;
            double deltaPhi_less = 0;
            if(iter == sortRecHits.begin())
            {
                cout << "Calculateing frist loop..." << endl;
                ConstMuonRecHitContainer::const_iterator iter_more = ++iter;
                --iter;
                ConstMuonRecHitContainer::const_iterator iter_less = sortRecHits.end();
                --iter_less;
                cout << "more_Phi: " << (*iter_more)->globalPosition().phi() << ", less_Phi: " << (*iter_less)->globalPosition().phi() << ", iter_Phi: " << (*iter)->globalPosition().phi() << endl;
                deltaPhi_more = (2 * M_PI) - ((*iter_more)->globalPosition().phi().value() - (*iter)->globalPosition().phi().value());
                deltaPhi_less = (*iter_less)->globalPosition().phi().value() - (*iter)->globalPosition().phi().value();
            }
            else if(iter == (--sortRecHits.end()))
            {
                cout << "Calculateing last loop..." << endl;
                ConstMuonRecHitContainer::const_iterator iter_less = --iter;
                ++iter;
                ConstMuonRecHitContainer::const_iterator iter_more = sortRecHits.begin();
                cout << "more_Phi: " << (*iter_more)->globalPosition().phi() << ", less_Phi: " << (*iter_less)->globalPosition().phi() << ", iter_Phi: " << (*iter)->globalPosition().phi() << endl;
                deltaPhi_less = (2 * M_PI) - ((*iter)->globalPosition().phi().value() - (*iter_less)->globalPosition().phi().value());
                deltaPhi_more = (*iter)->globalPosition().phi().value() - (*iter_more)->globalPosition().phi().value();
            }
            else
            {
                cout << "Calculateing " << n << "st loop..." << endl;
                ConstMuonRecHitContainer::const_iterator iter_less = --iter;
                ++iter;
                ConstMuonRecHitContainer::const_iterator iter_more = ++iter;
                --iter;
                cout << "more_Phi: " << (*iter_more)->globalPosition().phi() << ", less_Phi: " << (*iter_less)->globalPosition().phi() << ", iter_Phi: " << (*iter)->globalPosition().phi() << endl;
                deltaPhi_less = (2 * M_PI) - ((*iter)->globalPosition().phi().value() - (*iter_less)->globalPosition().phi().value());
                deltaPhi_more = (2 * M_PI) - ((*iter_more)->globalPosition().phi().value() - (*iter)->globalPosition().phi().value());
            }
            if(deltaPhi > deltaPhi_more)
                deltaPhi = deltaPhi_more;
            if(deltaPhi > deltaPhi_less)
                deltaPhi = deltaPhi_less;

            cout << "For this loop deltaPhi_more is " << deltaPhi_more << endl;
            cout << "For this loop deltaPhi_less is " << deltaPhi_less << endl;
            cout << "For this loop deltaPhi is " << deltaPhi << endl;
        }
        return deltaPhi;
    }
}

void RPCSeedrecHitFinder::checkandfill() {

    if(therecHits.size() >= 3)
    {
        theSeed->setrecHits(therecHits); 
        theSeed->seed();
    }
    else
        cout << "Layer less than 3, could not fill a RPCSeedFinder" << endl;
}
