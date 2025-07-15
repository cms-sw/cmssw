// Node.h

#ifndef L1Trigger_L1TMuonEndCap_emtf_Node
#define L1Trigger_L1TMuonEndCap_emtf_Node

#include <string>
#include <vector>
#include <memory>
#include "Event.h"

namespace emtf {

  class Node {
  public:
    Node();
    Node(std::string cName);
    ~Node() = default;

    Node(Node &&) = default;
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    std::string getName() const;
    void setName(std::string sName);

    double getErrorReduction() const;
    void setErrorReduction(double sErrorReduction);

    Node *getLeftDaughter();
    const Node *getLeftDaughter() const;
    void setLeftDaughter(std::unique_ptr<Node> sLeftDaughter);

    const Node *getRightDaughter() const;
    Node *getRightDaughter();
    void setRightDaughter(std::unique_ptr<Node> sLeftDaughter);

    Node *getParent();
    const Node *getParent() const;
    void setParent(Node *sParent);

    double getSplitValue() const;
    void setSplitValue(double sSplitValue);

    int getSplitVariable() const;
    void setSplitVariable(int sSplitVar);

    double getFitValue() const;
    void setFitValue(double sFitValue);

    double getTotalError() const;
    void setTotalError(double sTotalError);

    double getAvgError() const;
    void setAvgError(double sAvgError);

    int getNumEvents() const;
    void setNumEvents(int sNumEvents);

    std::vector<std::vector<Event *> > &getEvents();
    void setEvents(std::vector<std::vector<Event *> > &sEvents);

    void calcOptimumSplit();
    void filterEventsToDaughters();
    Node *filterEventToDaughter(Event *e);
    void listEvents();
    void theMiracleOfChildBirth();

  private:
    std::string name;

    std::unique_ptr<Node> leftDaughter;
    std::unique_ptr<Node> rightDaughter;
    Node *parent;

    double splitValue;
    int splitVariable;

    double errorReduction;
    double totalError;
    double avgError;

    double fitValue;
    int numEvents;

    std::vector<std::vector<Event *> > events;
  };

}  // namespace emtf

#endif
