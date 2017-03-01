#include "L1Trigger/L1TGlobal/interface/GlobalParamsHelper.h"

using namespace l1t;


const GlobalParamsHelper *  GlobalParamsHelper::readFromEventSetup(const L1TGlobalParameters * es){
  return new GlobalParamsHelper(es);
}

GlobalParamsHelper *  GlobalParamsHelper::readAndWriteFromEventSetup(const L1TGlobalParameters * es){
  GlobalParamsHelper * x = new GlobalParamsHelper(es);
  x->useCopy();
  return x;
}

GlobalParamsHelper::GlobalParamsHelper(L1TGlobalParameters * w) {
  write_ = w; 
  check_write(); 
  we_own_write_ = false;
  write_->m_version = VERSION; 
  read_ = write_; 
}

GlobalParamsHelper::GlobalParamsHelper(const L1TGlobalParameters * es) {read_ = es; write_=NULL;}

void GlobalParamsHelper::useCopy(){
  write_ = new L1TGlobalParameters(*read_);
  we_own_write_ = true;
  read_  = write_;
}

GlobalParamsHelper::~GlobalParamsHelper() {
  if (we_own_write_ && write_) delete write_;
}

// set the number of bx in event
void GlobalParamsHelper::setTotalBxInEvent(
    const int& numberBxValue) {

    check_write();  write_->m_totalBxInEvent = numberBxValue;

}

// set the number of physics trigger algorithms
void GlobalParamsHelper::setNumberPhysTriggers(
    const unsigned int& numberPhysTriggersValue) {

    check_write();  write_->m_numberPhysTriggers = numberPhysTriggersValue;

}

// set the number of L1 muons received by GT
void GlobalParamsHelper::setNumberL1Mu(const unsigned int& numberL1MuValue) {

    check_write();  write_->m_numberL1Mu = numberL1MuValue;

}

//  set the number of L1 e/gamma objects received by GT
void GlobalParamsHelper::setNumberL1EG(
    const unsigned int& numberL1EGValue) {

    check_write();  write_->m_numberL1EG = numberL1EGValue;

}

// set the number of L1 central jets received by GT
void GlobalParamsHelper::setNumberL1Jet(
    const unsigned int& numberL1JetValue) {

    check_write();  write_->m_numberL1Jet = numberL1JetValue;

}

// set the number of L1 tau jets received by GT
void GlobalParamsHelper::setNumberL1Tau(
    const unsigned int& numberL1TauValue) {

    check_write();  write_->m_numberL1Tau = numberL1TauValue;

}

// hardware stuff

// set the number of condition chips in GTL
void GlobalParamsHelper::setNumberChips(
    const unsigned int& numberChipsValue) {

    check_write();  write_->m_numberChips = numberChipsValue;

}

// set the number of pins on the GTL condition chips
void GlobalParamsHelper::setPinsOnChip(
    const unsigned int& pinsOnChipValue) {

    check_write();  write_->m_pinsOnChip = pinsOnChipValue;

}

// set the correspondence "condition chip - GTL algorithm word"
// in the hardware
void GlobalParamsHelper::setOrderOfChip(
    const std::vector<int>& orderOfChipValue) {

    check_write();  write_->m_orderOfChip = orderOfChipValue;

}

// print all the L1 GT stable parameters
void GlobalParamsHelper::print(std::ostream& myStr) const {
    myStr << "\nL1T Global  Parameters \n" << std::endl;


    // number of bx
    myStr << "\n  Number of bx in Event =            "
        << read_->m_totalBxInEvent << std::endl;

    // trigger decision

    // number of physics trigger algorithms
    myStr << "\n  Number of physics trigger algorithms =            "
        << read_->m_numberPhysTriggers << std::endl;

    // muons
    myStr << "\n  Number of muons received by L1 GT =                     "
        << read_->m_numberL1Mu << std::endl;

    // e/gamma and isolated e/gamma objects
    myStr << "  Number of e/gamma objects received by L1 GT =          "
        << read_->m_numberL1EG << std::endl;

    // central, forward and tau jets
    myStr << "\n  Number of  jets received by L1 GT =             "
        << read_->m_numberL1Jet << std::endl;

    myStr << "  Number of tau  received by L1 GT =                 "
        << read_->m_numberL1Tau << std::endl;


    // hardware

    // number of condition chips
    myStr << "\n  Number of condition chips =                        "
        << read_->m_numberChips << std::endl;

    // number of pins on the GTL condition chips
    myStr << "  Number of pins on chips =        "
        << read_->m_pinsOnChip << std::endl;

    // correspondence "condition chip - GTL algorithm word" in the hardware
    // chip 2: 0 - 95;  chip 1: 96 - 128 (191)
    myStr << "  Order of  chips for algorithm word = {";

    for (unsigned int iChip = 0; iChip < read_->m_orderOfChip.size(); ++iChip) {
        myStr << read_->m_orderOfChip[iChip];
        if (iChip != (read_->m_orderOfChip.size() - 1)) {
            myStr << ", ";
        }
    }

    myStr << "}" << std::endl;

    myStr << "\n" << std::endl;

}
