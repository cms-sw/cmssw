#include "DataFormats/L1TParticleFlow/interface/jets.h"

const std::unordered_map<std::string, l1ct::io_v1::JetTagClass::JetTagClassValue> l1ct::io_v1::JetTagClass::labels_ = {
    {"b", l1ct::io_v1::JetTagClass::JetTagClassValue::b},
    {"c", l1ct::io_v1::JetTagClass::JetTagClassValue::c},
    {"uds", l1ct::io_v1::JetTagClass::JetTagClassValue::uds},
    {"g", l1ct::io_v1::JetTagClass::JetTagClassValue::g},
    {"tau_p", l1ct::io_v1::JetTagClass::JetTagClassValue::tau_p},
    {"tau_n", l1ct::io_v1::JetTagClass::JetTagClassValue::tau_n},
    {"mu", l1ct::io_v1::JetTagClass::JetTagClassValue::mu},
    {"e", l1ct::io_v1::JetTagClass::JetTagClassValue::e}};

const l1ct::io_v1::JetTagClass l1ct::io_v1::JetTagClassHandler::tagClassesDefault_[NTagFields] = {
    l1ct::io_v1::JetTagClass("b"),
    l1ct::io_v1::JetTagClass("c"),
    l1ct::io_v1::JetTagClass("uds"),
    l1ct::io_v1::JetTagClass("g"),
    l1ct::io_v1::JetTagClass("tau_p"),
    l1ct::io_v1::JetTagClass("tau_n"),
    l1ct::io_v1::JetTagClass("mu"),
    l1ct::io_v1::JetTagClass("e")};
