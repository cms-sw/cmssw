#include "DataFormats/L1TParticleFlow/interface/jets.h"

const std::unordered_map<std::string, l1ct::JetTagClass::JetTagClassValue> l1ct::JetTagClass::labels_ = {
  {"uds"   , l1ct::JetTagClass::JetTagClassValue::uds},
  {"g"     , l1ct::JetTagClass::JetTagClassValue::g},
  {"b"     , l1ct::JetTagClass::JetTagClassValue::b},
  {"c"     , l1ct::JetTagClass::JetTagClassValue::c},
  {"tau_p" , l1ct::JetTagClass::JetTagClassValue::tau_p},
  {"tau_n" , l1ct::JetTagClass::JetTagClassValue::tau_n},
  {"e"     , l1ct::JetTagClass::JetTagClassValue::e},
  {"mu"    , l1ct::JetTagClass::JetTagClassValue::mu}
};

const l1ct::JetTagClass l1ct::Jet::tagClassesDefault_[NTagFields]= {l1ct::JetTagClass("b"),
                                                                    l1ct::JetTagClass("c"),
                                                                    l1ct::JetTagClass("uds"),
                                                                    l1ct::JetTagClass("g"),
                                                                    l1ct::JetTagClass("tau_p"),
                                                                    l1ct::JetTagClass("tau_n"),
                                                                    l1ct::JetTagClass("e"),
                                                                    l1ct::JetTagClass("mu")};
