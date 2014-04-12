#include "RecoParticleFlow/PFProducer/interface/PFAlgoTestBenchConversions.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

using namespace std;
using namespace reco;

 
void PFAlgoTestBenchConversions::processBlock(const reco::PFBlockRef& blockref,
					    std::list<PFBlockRef>& hcalBlockRefs,
					    std::list<PFBlockRef>& ecalBlockRefs)
{

  cout<<"conversions test bench: process block"
      <<(*blockref)<<endl;
}


