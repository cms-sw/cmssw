#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"

L1GctJetFinalStage::L1GctJetFinalStage()
{
}

L1GctJetFinalStage::~L1GctJetFinalStage()
{
}

void L1GctJetFinalStage::reset()
{
}

void L1GctJetFinalStage::fetchInput()
{
}

void L1GctJetFinalStage::process()
{
}

void L1GctJetFinalStage::setInputWheelJetFpga(int i, L1GctWheelJetFpga* wjf)
{
    assert(i >= 0);  //&& < some max number of wheelJetFpgas
    m_wheelFpgas[i] = wjf;   
}

void L1GctJetFinalStage::setInputJet(int i, L1GctJet jet)
{
}
