#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/TypeMatch.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "DataFormats/Common/interface/PathStatus.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <boost/spirit/include/qi.hpp>

#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"

#include <string>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include <tuple>

/** pathStatusExpression is borrowed from \class edm::PathStatusFilter by W. David Dagenhart */

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;

namespace pathStatusExpression {
  class Evaluator {
  public:
    virtual ~Evaluator() {}

    enum EvaluatorType { Name, Not, And, Or, BeginParen };
    virtual EvaluatorType type() const = 0;

    virtual const char* pathName() const { return ""; }

    virtual void setLeft(std::unique_ptr<Evaluator>&&) {}
    virtual void setRight(std::unique_ptr<Evaluator>&&) {}

    virtual void print(std::ostream& out, unsigned int indentation) const {}
    virtual void init(edm::ConsumesCollector&) {}
    virtual bool evaluate(edm::Event const& event) const { return true; };
  };

  class Operand : public Evaluator {
  public:
    Operand(std::vector<char> const& pathName) : pathName_(pathName.begin(), pathName.end()) {}

    EvaluatorType type() const override { return Name; }

    void print(std::ostream& out, unsigned int indentation) const override {
      out << std::string(indentation, ' ') << pathName_ << "\n";
    }

    void init(edm::ConsumesCollector& iC) override { token_ = iC.consumes<edm::PathStatus>(edm::InputTag(pathName_)); }

    bool evaluate(edm::Event const& event) const override { return event.get(token_).accept(); }

  private:
    std::string pathName_;
    edm::EDGetTokenT<edm::PathStatus> token_;
  };

  class NotOperator : public Evaluator {
  public:
    EvaluatorType type() const override { return Not; }

    void setLeft(std::unique_ptr<Evaluator>&& v) override { operand_ = std::move(v); }

    void print(std::ostream& out, unsigned int indentation) const override {
      out << std::string(indentation, ' ') << "not\n";
      operand_->print(out, indentation + 4);
    }

    void init(edm::ConsumesCollector& iC) override { operand_->init(iC); }

    bool evaluate(edm::Event const& event) const override { return !operand_->evaluate(event); }

  private:
    edm::propagate_const<std::unique_ptr<Evaluator>> operand_;
  };

  template <typename T>
  class BinaryOperator : public Evaluator {
  public:
    EvaluatorType type() const override;

    void setLeft(std::unique_ptr<Evaluator>&& v) override { left_ = std::move(v); }
    void setRight(std::unique_ptr<Evaluator>&& v) override { right_ = std::move(v); }

    void print(std::ostream& out, unsigned int indentation) const override;

    void init(edm::ConsumesCollector& iC) override {
      left_->init(iC);
      right_->init(iC);
    }

    bool evaluate(edm::Event const& event) const override {
      T op;
      return op(left_->evaluate(event), right_->evaluate(event));
    }

  private:
    edm::propagate_const<std::unique_ptr<Evaluator>> left_;
    edm::propagate_const<std::unique_ptr<Evaluator>> right_;
  };

  template <>
  inline Evaluator::EvaluatorType BinaryOperator<std::logical_and<bool>>::type() const {
    return And;
  }

  template <>
  inline Evaluator::EvaluatorType BinaryOperator<std::logical_or<bool>>::type() const {
    return Or;
  }

  template <>
  void BinaryOperator<std::logical_and<bool>>::print(std::ostream& out, unsigned int indentation) const {
    out << std::string(indentation, ' ') << "and\n";
    left_->print(out, indentation + 4);
    right_->print(out, indentation + 4);
  }
  template <>
  void BinaryOperator<std::logical_or<bool>>::print(std::ostream& out, unsigned int indentation) const {
    out << std::string(indentation, ' ') << "or\n";
    left_->print(out, indentation + 4);
    right_->print(out, indentation + 4);
  }

  using AndOperator = BinaryOperator<std::logical_and<bool>>;
  using OrOperator = BinaryOperator<std::logical_or<bool>>;

  class BeginParenthesis : public Evaluator {
  public:
    EvaluatorType type() const override { return BeginParen; }
  };

  // This class exists to properly handle the precedence of the
  // operators and also handle the order of operations specified
  // by parentheses. (search for shunting yard algorithm on the
  // internet for a description of this algorithm)
  class ShuntingYardAlgorithm {
  public:
    void addPathName(std::vector<char> const& s) {
      operandStack.push_back(std::make_unique<Operand>(s));
      pathNames_.emplace_back(s.begin(), s.end());
    }

    void addOperatorNot() {
      if (operatorStack.empty() || operatorStack.back()->type() != Evaluator::Not) {
        operatorStack.push_back(std::make_unique<NotOperator>());
      } else {
        // Two Not operations in a row cancel and are the same as no operation at all.
        operatorStack.pop_back();
      }
    }

    void moveBinaryOperator() {
      std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
      backEvaluator->setRight(std::move(operandStack.back()));
      operandStack.pop_back();
      backEvaluator->setLeft(std::move(operandStack.back()));
      operandStack.pop_back();
      operandStack.push_back(std::move(backEvaluator));
      operatorStack.pop_back();
    }

    void moveNotOperator() {
      std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
      backEvaluator->setLeft(std::move(operandStack.back()));
      operandStack.pop_back();
      operandStack.push_back(std::move(backEvaluator));
      operatorStack.pop_back();
    }

    void addOperatorAnd() {
      while (!operatorStack.empty()) {
        std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
        if (backEvaluator->type() == Evaluator::And) {
          moveBinaryOperator();
        } else if (backEvaluator->type() == Evaluator::Not) {
          moveNotOperator();
        } else {
          break;
        }
      }
      operatorStack.push_back(std::make_unique<AndOperator>());
    }

    void addOperatorOr() {
      while (!operatorStack.empty()) {
        std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
        if (backEvaluator->type() == Evaluator::And || backEvaluator->type() == Evaluator::Or) {
          moveBinaryOperator();
        } else if (backEvaluator->type() == Evaluator::Not) {
          moveNotOperator();
        } else {
          break;
        }
      }
      operatorStack.push_back(std::make_unique<OrOperator>());
    }

    void addBeginParenthesis() { operatorStack.push_back(std::make_unique<BeginParenthesis>()); }

    void addEndParenthesis() {
      while (!operatorStack.empty()) {
        std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
        if (backEvaluator->type() == Evaluator::BeginParen) {
          operatorStack.pop_back();
          break;
        }
        if (backEvaluator->type() == Evaluator::And || backEvaluator->type() == Evaluator::Or) {
          moveBinaryOperator();
        } else if (backEvaluator->type() == Evaluator::Not) {
          moveNotOperator();
        }
      }
    }

    std::unique_ptr<Evaluator> finish() {
      while (!operatorStack.empty()) {
        std::unique_ptr<Evaluator>& backEvaluator = operatorStack.back();
        // Just a sanity check. The grammar defined for the boost Spirit parser
        // should catch any errors of this type before we get here.
        if (backEvaluator->type() == Evaluator::BeginParen) {
          throw cms::Exception("LogicError") << "Should be impossible to get this error. Contact a Framework developer";
        }
        if (backEvaluator->type() == Evaluator::And || backEvaluator->type() == Evaluator::Or) {
          moveBinaryOperator();
        } else if (backEvaluator->type() == Evaluator::Not) {
          moveNotOperator();
        }
      }
      // Just a sanity check. The grammar defined for the boost Spirit parser
      // should catch any errors of this type before we get here.
      if (!operatorStack.empty() || operandStack.size() != 1U) {
        throw cms::Exception("LogicError") << "Should be impossible to get this error. Contact a Framework developer";
      }
      std::unique_ptr<Evaluator> temp = std::move(operandStack.back());
      operandStack.pop_back();
      return temp;
    }

    const std::vector<std::string>& pathNames() { return pathNames_; }

  private:
    std::vector<std::string> pathNames_;
    std::vector<std::unique_ptr<Evaluator>> operandStack;
    std::vector<std::unique_ptr<Evaluator>> operatorStack;
  };

  // Use boost Spirit to parse the logical expression character string
  template <typename Iterator>
  class Grammar : public qi::grammar<Iterator, ascii::space_type> {
  public:
    Grammar(ShuntingYardAlgorithm* algorithm) : Grammar::base_type(expression), algorithm_(algorithm) {
      // setup functors that call into shunting algorithm while parsing the logical expression
      auto addPathName = std::bind(&ShuntingYardAlgorithm::addPathName, algorithm_, std::placeholders::_1);
      auto addOperatorNot = std::bind(&ShuntingYardAlgorithm::addOperatorNot, algorithm_);
      auto addOperatorAnd = std::bind(&ShuntingYardAlgorithm::addOperatorAnd, algorithm_);
      auto addOperatorOr = std::bind(&ShuntingYardAlgorithm::addOperatorOr, algorithm_);
      auto addBeginParenthesis = std::bind(&ShuntingYardAlgorithm::addBeginParenthesis, algorithm_);
      auto addEndParenthesis = std::bind(&ShuntingYardAlgorithm::addEndParenthesis, algorithm_);

      // Define the syntax allowed in the logical expressions
      pathName = !unaryOperator >> !binaryOperatorTest >> (+qi::char_("a-zA-Z0-9_"))[addPathName];
      binaryOperand = (qi::lit('(')[addBeginParenthesis] >> expression >> qi::lit(')')[addEndParenthesis]) |
                      (unaryOperator[addOperatorNot] >> binaryOperand) | pathName;
      afterOperator = ascii::space | &qi::lit('(') | &qi::eoi;
      unaryOperator = qi::lit("not") >> afterOperator;
      // The only difference in the next two is that one calls a functor and the other does not
      binaryOperatorTest = (qi::lit("and") >> afterOperator) | (qi::lit("or") >> afterOperator);
      binaryOperator =
          (qi::lit("and") >> afterOperator)[addOperatorAnd] | (qi::lit("or") >> afterOperator)[addOperatorOr];
      expression = binaryOperand % binaryOperator;
    }

  private:
    qi::rule<Iterator> pathName;
    qi::rule<Iterator, ascii::space_type> binaryOperand;
    qi::rule<Iterator> afterOperator;
    qi::rule<Iterator> unaryOperator;
    qi::rule<Iterator> binaryOperatorTest;
    qi::rule<Iterator> binaryOperator;
    qi::rule<Iterator, ascii::space_type> expression;

    ShuntingYardAlgorithm* algorithm_;
  };
}  // namespace pathStatusExpression

using namespace l1t;

class L1GTAlgoBlockProducer
    : public edm::one::EDProducer<edm::RunCache<std::unordered_map<std::string, unsigned int>>> {
public:
  explicit L1GTAlgoBlockProducer(const edm::ParameterSet&);
  ~L1GTAlgoBlockProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  struct AlgoDefinition {
    edm::propagate_const<std::unique_ptr<pathStatusExpression::Evaluator>> evaluator_;
    std::vector<std::string> pathNames_;
    std::set<std::tuple<std::string, std::string>> filtModules_;
    unsigned int prescale_;
    unsigned int prescalePreview_;
    std::set<unsigned int> bunchMask_;
    bool isVeto_;
    int triggerTypes_;
  };

  void init(const edm::ProcessHistory&);
  void produce(edm::Event&, const edm::EventSetup&) override;
  std::shared_ptr<std::unordered_map<std::string, unsigned int>> globalBeginRun(edm::Run const&,
                                                                                edm::EventSetup const&) const override;

  void globalEndRun(edm::Run const&, edm::EventSetup const&) {}

  edm::GetterOfProducts<P2GTCandidateVectorRef> getterOfPassedReferences_;
  std::map<std::string, AlgoDefinition> algoDefinitions_;
  bool initialized_ = false;
  int bunchCrossingEmu_ = 0;
};

void L1GTAlgoBlockProducer::fillDescriptions(edm::ConfigurationDescriptions& description) {
  edm::ParameterSetDescription algoDesc;
  algoDesc.add<std::string>("name", "");
  algoDesc.add<std::string>("expression");
  algoDesc.add<double>("prescale", 1);
  algoDesc.add<double>("prescalePreview", 1);
  algoDesc.add<std::vector<unsigned int>>("bunchMask", {});

  algoDesc.add<std::vector<int>>("triggerTypes", {1});
  algoDesc.add<bool>("veto", false);

  edm::ParameterSetDescription desc;
  desc.addVPSet("algorithms", algoDesc);

  description.addWithDefaultLabel(desc);
}

L1GTAlgoBlockProducer::L1GTAlgoBlockProducer(const edm::ParameterSet& config)
    : getterOfPassedReferences_(edm::TypeMatch(), this) {
  edm::ConsumesCollector iC(consumesCollector());

  for (const auto& algoConfig : config.getParameterSetVector("algorithms")) {
    const std::string logicalExpression = algoConfig.getParameter<std::string>("expression");
    std::string name = algoConfig.getParameter<std::string>("name");
    if (name.empty()) {
      name = logicalExpression;
    }

    pathStatusExpression::ShuntingYardAlgorithm shuntingYardAlgorithm;
    pathStatusExpression::Grammar<std::string::const_iterator> grammar(&shuntingYardAlgorithm);

    auto it = logicalExpression.cbegin();
    if (!qi::phrase_parse(it, logicalExpression.cend(), grammar, ascii::space) || (it != logicalExpression.cend())) {
      throw cms::Exception("Configuration") << "Syntax error in logical expression. Here is an example of how\n"
                                            << "the syntax should look:\n"
                                            << "    \"path1 and not (path2 or not path3)\"\n"
                                            << "The expression must contain alternating appearances of operands\n"
                                            << "which are path names and binary operators which can be \'and\'\n"
                                            << "or \'or\', with a path name at the beginning and end. There\n"
                                            << "must be at least one path name. In addition to the alternating\n"
                                            << "path names and binary operators, the unary operator \'not\' can\n"
                                            << "be inserted before a path name or a begin parenthesis.\n"
                                            << "Parentheses are allowed. Parentheses must come in matching pairs.\n"
                                            << "Matching begin and end parentheses must contain a complete and\n"
                                            << "syntactically correct logical expression. There must be at least\n"
                                            << "one space or parenthesis between operators and path names. Extra\n"
                                            << "space is ignored and OK. Path names can only contain upper and\n"
                                            << "lower case letters, numbers, and underscores. A path name cannot\n"
                                            << "be the same as an operator name.\n";
    }

    AlgoDefinition definition;

    for (const std::string& pathName : shuntingYardAlgorithm.pathNames()) {
      definition.pathNames_.push_back(pathName);
    }

    definition.evaluator_ = shuntingYardAlgorithm.finish();
    double dPrescale = algoConfig.getParameter<double>("prescale");
    if ((dPrescale < 1. || dPrescale >= std::pow(2, 24) / 100 - 1) && dPrescale != 0) {
      throw cms::Exception("Configuration")
          << "Prescale value error. Expected prescale value between 1 and 2^24/100 - 1 or 0 got " << dPrescale << "."
          << std::endl;
    }

    definition.prescale_ = std::round(dPrescale * 100);

    double dPrescalePreview = algoConfig.getParameter<double>("prescalePreview");
    if ((dPrescalePreview < 1. || dPrescalePreview >= std::pow(2, 24) / 100 - 1) && dPrescalePreview != 0) {
      throw cms::Exception("Configuration")
          << "PrescalePreview value error. Expected prescale value between 1 and 2^24/100 - 1 or 0 got "
          << dPrescalePreview << "." << std::endl;
    }

    definition.prescalePreview_ = std::round(dPrescalePreview * 100);

    std::vector<unsigned int> vBunchMask = algoConfig.getParameter<std::vector<unsigned int>>("bunchMask");
    definition.bunchMask_ =
        std::set<unsigned int>(std::make_move_iterator(vBunchMask.begin()), std::make_move_iterator(vBunchMask.end()));

    definition.isVeto_ = algoConfig.getParameter<bool>("veto");
    definition.triggerTypes_ = 0;
    const std::vector<int> triggerTypes = algoConfig.getParameter<std::vector<int>>("triggerTypes");
    for (int type : triggerTypes) {
      definition.triggerTypes_ |= type;
    }

    definition.evaluator_->init(iC);
    auto [iter, wasInserted] = algoDefinitions_.emplace(std::move(name), std::move(definition));
    if (!wasInserted) {
      throw cms::Exception("Configuration")
          << "Algorithm " << iter->first << " already exists. Algorithm names must be unique." << std::endl;
    }
  }

  callWhenNewProductsRegistered(getterOfPassedReferences_);
  produces<P2GTAlgoBlockMap>();
}

void L1GTAlgoBlockProducer::init(const edm::ProcessHistory& pHistory) {
  const std::string& pName = edm::Service<edm::service::TriggerNamesService>()->getProcessName();

  edm::ProcessConfiguration cfg;

  pHistory.getConfigurationForProcess(pName, cfg);

  const edm::ParameterSet* pset = edm::pset::Registry::instance()->getMapped(cfg.parameterSetID());

  for (auto& [name, algoDef] : algoDefinitions_) {
    for (const std::string& pathName : algoDef.pathNames_) {
      if (pset->existsAs<std::vector<std::string>>(pathName)) {
        const auto& modules = pset->getParameter<std::vector<std::string>>(pathName);
        for (const auto& mod : modules) {
          if (mod.front() != std::string("-") && pset->exists(mod)) {
            const auto& modPSet = pset->getParameterSet(mod);
            if (modPSet.getParameter<std::string>("@module_edm_type") == "EDFilter") {
              if (modPSet.getParameter<std::string>("@module_type") == "L1GTSingleObjectCond") {
                algoDef.filtModules_.insert({mod, modPSet.getParameter<edm::InputTag>("tag").instance()});
              } else if (modPSet.getParameter<std::string>("@module_type") == "L1GTDoubleObjectCond") {
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection1").getParameter<edm::InputTag>("tag").instance()});
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection2").getParameter<edm::InputTag>("tag").instance()});
              } else if (modPSet.getParameter<std::string>("@module_type") == "L1GTTripleObjectCond") {
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection1").getParameter<edm::InputTag>("tag").instance()});
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection2").getParameter<edm::InputTag>("tag").instance()});
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection3").getParameter<edm::InputTag>("tag").instance()});
              } else if (modPSet.getParameter<std::string>("@module_type") == "L1GTQuadObjectCond") {
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection1").getParameter<edm::InputTag>("tag").instance()});
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection2").getParameter<edm::InputTag>("tag").instance()});
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection3").getParameter<edm::InputTag>("tag").instance()});
                algoDef.filtModules_.insert(
                    {mod, modPSet.getParameterSet("collection4").getParameter<edm::InputTag>("tag").instance()});
              }
            }
          }
        }
      }
    }
  }
}

std::shared_ptr<std::unordered_map<std::string, unsigned int>> L1GTAlgoBlockProducer::globalBeginRun(
    edm::Run const&, edm::EventSetup const&) const {
  // Reset prescale counters at the beginning of a new run
  return std::make_shared<std::unordered_map<std::string, unsigned int>>();
}

void L1GTAlgoBlockProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  if (!initialized_) {
    init(event.processHistory());
    initialized_ = true;
  }

  std::vector<edm::Handle<P2GTCandidateVectorRef>> handles;
  getterOfPassedReferences_.fillHandles(event, handles);

  std::unique_ptr<P2GTAlgoBlockMap> algoCollection = std::make_unique<P2GTAlgoBlockMap>();

  std::unordered_map<std::string, unsigned int>& prescaleCounters = *runCache(event.getRun().index());

  int bunchCrossing = event.isRealData() ? event.bunchCrossing() : bunchCrossingEmu_ + 1;

  for (const auto& [name, algoDef] : algoDefinitions_) {
    bool decisionBeforeBxAndPrescale = algoDef.evaluator_->evaluate(event);
    bool decisionBeforePrescale = decisionBeforeBxAndPrescale && algoDef.bunchMask_.count(bunchCrossing) == 0;
    bool decisionFinal = false;
    bool decisionFinalPreview = false;

    if (algoDef.prescale_ != 0 && decisionBeforePrescale) {
      if (prescaleCounters.count(name) > 0) {
        prescaleCounters[name] += 100;
      } else {
        prescaleCounters[name] = 100;
      }

      if (prescaleCounters[name] >= algoDef.prescale_) {
        decisionFinal = true;
        prescaleCounters[name] -= algoDef.prescale_;
      } else {
        decisionFinal = false;
      }
    }

    if (algoDef.prescalePreview_ != 0 && decisionBeforePrescale) {
      std::string previewName = name + "_preview";
      if (prescaleCounters.count(previewName) > 0) {
        prescaleCounters[previewName] += 100;
      } else {
        prescaleCounters[previewName] = 100;
      }

      if (prescaleCounters[previewName] >= algoDef.prescalePreview_) {
        decisionFinalPreview = true;
        prescaleCounters[previewName] -= algoDef.prescalePreview_;
      } else {
        decisionFinalPreview = false;
      }
    }

    P2GTCandidateVectorRef trigObjects;

    if (decisionFinal) {
      for (const auto& handle : handles) {
        const std::string& module = handle.provenance()->moduleLabel();
        const std::string& instance = handle.provenance()->productInstanceName();

        if (algoDef.filtModules_.count({module, instance}) > 0) {
          trigObjects.insert(trigObjects.end(), handle->begin(), handle->end());
        }
      }
    }

    algoCollection->emplace(name,
                            P2GTAlgoBlock(decisionBeforeBxAndPrescale,
                                          decisionBeforePrescale,
                                          decisionFinal,
                                          decisionFinalPreview,
                                          algoDef.isVeto_,
                                          algoDef.triggerTypes_,
                                          std::move(trigObjects)));
  }

  event.put(std::move(algoCollection));

  bunchCrossingEmu_ = (bunchCrossingEmu_ + 1) % 3564;
}

DEFINE_FWK_MODULE(L1GTAlgoBlockProducer);
