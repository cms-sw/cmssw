public:

int path() const {
  int p(-2);
  edm::CurrentProcessingContext const* cpc(currentContext());
  if (cpc!=0) {
    p=cpc->pathInSchedule();
  }
  return p;
}

int module() const {
  int m(-2);
  edm::CurrentProcessingContext const* cpc(currentContext());
  if (cpc!=0) {
    m=cpc->slotInPath();
  }
  return m;
}

std::pair<int,int> pmid() const {
  std::pair<int,int>pm(-2,-2);
  edm::CurrentProcessingContext const* cpc(currentContext());
  if (cpc!=0) {
    pm.first =cpc->pathInSchedule();
    pm.second=cpc->slotInPath();
  }
  return pm;
}

/*
const std::string* pathName() const {
  edm::CurrentProcessingContext const* cpc(currentContext());
  if (cpc!=0) {
    return cpc->pathName();
  } else {
    return 0;
  }
}

const std::string* moduleLabel() const {
  edm::CurrentProcessingContext const* cpc(currentContext());
  if (cpc!=0) {
    return cpc->moduleLabel();
  } else {
    return 0;
  }
}
*/
