// Use something like the following to run this file:
//
//   root.exe -b -l -q SchemaEvolutionTest.root 'IOPool/Input/test/testForStreamerInfo.C(gFile)' | sort -u
//
// In the output, lines with "Missing" indicate problems.

// Note the script ignores classes whose name start with "T"
// to ignore ROOT classes. In the context this is used, none
// of the interesting classes start with "T" (although if used
// generally use I can imagine this could cause some confusion...).

// This is a modified version of some temporary code Philippe Canal
// provided us while debugging a problem.

#include "TFile.h"
#include "TStreamerInfo.h"
#include "TList.h"
#include "TVirtualCollectionProxy.h"
#include "TStreamerElement.h"
#include <iostream>

void check(TClass &cl, TList &streamerInfoList) {
  std::string name(cl.GetName());
  if (name == "string")
    return;
  if (0 == name.compare(0, strlen("pair<"), "pair<"))
    return;
  //std::cout << "check TClass: " << name << std::endl;
  bool found = streamerInfoList.FindObject(cl.GetName()) != 0;
  if (!found)
    std::cout << "Missing: " << cl.GetName() << '\n';
}

void check(TVirtualCollectionProxy &proxy, TList &streamerInfoList) {
  auto inner = proxy.GetValueClass();
  if (inner) {
    auto subproxy = inner->GetCollectionProxy();
    if (subproxy) {
      check(*subproxy, streamerInfoList);
    } else {
      check(*inner, streamerInfoList);
    }
  }
}

void check(TStreamerElement &element, TList &streamerInfoList) {
  auto cl = element.GetClass();
  if (cl == nullptr) {
    return;
  }
  // Ignore all classes that start with a T with the intent
  // to ignore all internal ROOT classes
  // (In general this might ignore other interesting classes,
  // but this is intended to be used in a specific test where the
  // the interesting classes don't start with T).
  if (*(cl->GetName()) == 'T') {
    return;
  }
  if (cl->GetCollectionProxy()) {
    check(*cl->GetCollectionProxy(), streamerInfoList);
  } else {
    check(*cl, streamerInfoList);
  }
}

// This is called once for each TStreamerInfo in
// streamerInfoList. The info is the first argument
// and the list of all the infos is the second argument.
void scan(TStreamerInfo &info, TList &streamerInfoList) {
  auto cl = TClass::GetClass(info.GetName());
  // print error message and do skip the info if there
  // is not a TClass available.
  if (!cl) {
    //std::cerr << "Error no TClass for " << info.GetName() << '\n';
    return;
  }
  auto proxy = cl->GetCollectionProxy();
  if (proxy)
    check(*proxy, streamerInfoList);
  for (auto e : TRangeDynCast<TStreamerElement>(*info.GetElements())) {
    if (!e)
      continue;
    check(*e, streamerInfoList);
  }
}

void scan(TList *streamerInfoList) {
  if (!streamerInfoList)
    return;
  for (auto l : TRangeDynCast<TStreamerInfo>(*streamerInfoList)) {
    if (!l)
      continue;
    // Ignore all classes that start with a T with the intent
    // to ignore all internal ROOT classes
    // (In general this might ignore other interesting classes,
    // but this is intended to be used in a specific test where the
    // the interesting classes don't start with T).
    if (*(l->GetName()) == 'T') {
      continue;
    }
    //std::cout << "Seeing: " << l->GetName() << " " << l->GetClassVersion() << '\n';
    scan(*l, *streamerInfoList);
  }
  delete streamerInfoList;
}

void testForStreamerInfo(TFile *file) {
  if (!file)
    return;
  scan(file->GetStreamerInfoList());
}
