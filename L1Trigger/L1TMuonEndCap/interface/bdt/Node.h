// Node.h

#ifndef L1Trigger_L1TMuonEndCap_emtf_Node
#define L1Trigger_L1TMuonEndCap_emtf_Node

#include <string>
#include <vector>
#include "Event.h"

namespace emtf {

  class Node {
  public:
    Node();
    Node(std::string cName);
    ~Node();

    Node(Node &&) = default;
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;

    std::string getName();
    void setName(std::string sName);

    double getErrorReduction();
    void setErrorReduction(double sErrorReduction);

    Node *getLeftDaughter();
    void setLeftDaughter(Node *sLeftDaughter);

    Node *getRightDaughter();
    void setRightDaughter(Node *sLeftDaughter);

    Node *getParent();
    void setParent(Node *sParent);

    double getSplitValue();
    void setSplitValue(double sSplitValue);

    int getSplitVariable();
    void setSplitVariable(int sSplitVar);

    double getFitValue();
    void setFitValue(double sFitValue);

    double getTotalError();
    void setTotalError(double sTotalError);

    double getAvgError();
    void setAvgError(double sAvgError);

    int getNumEvents();
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

    Node *leftDaughter;
    Node *rightDaughter;
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
