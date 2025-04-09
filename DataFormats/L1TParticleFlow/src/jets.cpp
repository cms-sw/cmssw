#include "DataFormats/L1TParticleFlow/interface/jets.h"

const std::unordered_map<std::string, l1ct::JetTagClass::JetTagClassValue> l1ct::JetTagClass::labels_ = {
    {"b", l1ct::JetTagClass::JetTagClassValue::b},
    {"c", l1ct::JetTagClass::JetTagClassValue::c},
    {"uds", l1ct::JetTagClass::JetTagClassValue::uds},
    {"g", l1ct::JetTagClass::JetTagClassValue::g},
    {"tau_p", l1ct::JetTagClass::JetTagClassValue::tau_p},
    {"tau_n", l1ct::JetTagClass::JetTagClassValue::tau_n},
    {"mu", l1ct::JetTagClass::JetTagClassValue::mu},
    {"e", l1ct::JetTagClass::JetTagClassValue::e}};

const l1ct::JetTagClass l1ct::JetTagClassHandler::tagClassesDefault_[NTagFields] = {l1ct::JetTagClass("b"),
                                                                                    l1ct::JetTagClass("c"),
                                                                                    l1ct::JetTagClass("uds"),
                                                                                    l1ct::JetTagClass("g"),
                                                                                    l1ct::JetTagClass("tau_p"),
                                                                                    l1ct::JetTagClass("tau_n"),
                                                                                    l1ct::JetTagClass("mu"),
                                                                                    l1ct::JetTagClass("e")};