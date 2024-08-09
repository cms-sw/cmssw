//  .
// ..: P. Chang, philip@physics.ucsd.edu

#include "stringutil.h"

//#############################################################################
// rstrip TString
//
void RooUtil::StringUtil::rstrip(TString &in, TString separator) {
  TString save(in);
  if (separator.EqualTo(" ")) {
    // Remove end-of-line spaces
    std::string str = in.Data();
    str.erase(str.find_last_not_of(" \n\r\t") + 1);
    in = str.c_str();
  } else {
    // Remove text after separator
    TObjArray *list = in.Tokenize(separator);
    if (list->GetEntries() != 0) {
      in = ((TObjString *)list->At(0))->GetString();
    }
  }
  // Print
  return;
}

//#############################################################################
// Convert TString -> vector<TString>
// like ' '.split()
//
RooUtil::StringUtil::vecTString RooUtil::StringUtil::split(TString in, TString separator) {
  RooUtil::StringUtil::vecTString out;
  TObjArray *list = in.Tokenize(separator);
  for (unsigned i = 0; i < (unsigned)list->GetEntries(); ++i) {
    TString token = ((TObjString *)list->At(i))->GetString();
    out.push_back(token);
  }
  if (out.size() == 0) {
    out.push_back("");
  }
  delete list;
  return out;
}

//#############################################################################
// Convert TString -> vector<TString>
// like ' '.rsplit()
//
RooUtil::StringUtil::vecTString RooUtil::StringUtil::rsplit(TString in, TString separator) {
  TString left = in;
  rstrip(left, separator);
  int size = left.Length();
  vecTString rtn;
  rtn.push_back(left);
  rtn.push_back(in(size + 1, in.Length() - size - 1));
  return rtn;
}

//#############################################################################
// Convert vector<TString> -> TString
// like ':'.join()
//
TString RooUtil::StringUtil::join(RooUtil::StringUtil::vecTString in, TString joiner, Int_t rm_blanks) {
  std::stringstream ss;
  for (unsigned i = 0; i < in.size(); ++i) {
    TString token = in[i];
    ss << token << ((i < in.size() - 1) ? joiner : "");
  }
  // Remove blanks
  TString out = ss.str();
  if (rm_blanks) {
    out.ReplaceAll(" ", "");
  }
  return out;
}

//#############################################################################
// Convert TString -> vector<TString> -> TString
//
TString RooUtil::StringUtil::sjoin(TString in, TString separator, TString joiner, Int_t rm_blanks) {
  RooUtil::StringUtil::vecTString vec = RooUtil::StringUtil::split(in, separator);
  TString out = RooUtil::StringUtil::join(vec, joiner, rm_blanks);
  return out;
}

//#############################################################################
RooUtil::StringUtil::vecTString RooUtil::StringUtil::filter(RooUtil::StringUtil::vecTString &vec, TString keyword) {
  RooUtil::StringUtil::vecTString newvec;
  for (unsigned i = 0; i < vec.size(); ++i) {
    if (!vec[i].Contains(keyword)) {
      continue;
    }
    newvec.push_back(vec[i]);
  }
  return newvec;
}

//#############################################################################
RooUtil::StringUtil::vecVecTString RooUtil::StringUtil::chunk(RooUtil::StringUtil::vecTString vec, Int_t nchunk) {
  int bunch_size = vec.size() / nchunk + (vec.size() % nchunk > 0);
  vecVecTString bunches;
  for (size_t i = 0; i < vec.size(); i += bunch_size) {
    auto last = std::min(vec.size(), i + bunch_size);
    bunches.emplace_back(vec.begin() + i, vec.begin() + last);
  }
  return bunches;
}

//#############################################################################
// From std::vector<TString> form an expression for TTree::Draw
TString RooUtil::StringUtil::formexpr(vecTString in) {
  in.erase(std::remove_if(in.begin(), in.end(), [](TString s) { return s.EqualTo("1"); }), in.end());
  if (in.size() == 0)
    in.push_back("1");
  return Form("(%s)", RooUtil::StringUtil::join(in, ")*(").Data());
}

//#############################################################################
// Clean unwanted parantheses
TString RooUtil::StringUtil::cleanparantheses(TString input) {
  std::string s = input.Data();
  remove_parantheses(s);
  return s.c_str();
}

//#############################################################################
// Under the hood for cleaning unwanted parantheses
void RooUtil::StringUtil::remove_parantheses(std::string &S) {
  using namespace std;
  map<int, bool> pmap;
  for (size_t i = 0; i < S.size(); i++) {
    map<int, bool>::iterator it;
    if (S.at(i) == '(') {
      pmap[i] = true;
    } else if (S.at(i) == ')') {
      it = pmap.end();
      it--;
      if (!(*it).second) {
        pmap.erase(it);
      } else {
        S.erase(S.begin() + i);
        S.erase(S.begin() + (*it).first);
        pmap.erase(it);
        i = i - 2;
      }
    } else {
      if (!pmap.empty()) {
        it = pmap.end();
        it--;
        (*it).second = false;
      }
    }
  }
}

//#############################################################################
// Given a template replace tokens by pattern.
// Could be thought of as "".format() from python. (although it's not nearly as good as that...)
TString RooUtil::StringUtil::format(TString tmp, std::vector<TString> tokens) {
  for (auto &token : tokens) {
    std::vector<TString> v = rsplit(token, "=");
    TString key = v[0];
    TString val = v[1];
    tmp.ReplaceAll(Form("{%s}", key.Data()), val);
  }
  return tmp;
}

//std::string RooUtil::StringUtil::parser(std::string _input, int loc_){
//
//    using namespace std;
//
//    string input = _input;
//
//    set<char> support;
//    support.insert('+');
//    support.insert('-');
//    support.insert('*');
//    support.insert('/');
//    support.insert('>');
//    support.insert('<');
//    support.insert('=');
//
//    string expi;
//    set<char> op;
//    int loc = loc_;
//    int size = input.size();
//
//    while(1){
//        if(input[loc] ==  '('){
//            expi += parser(input,loc+1);
//        }else if(input[loc] == ')'){
//          if((input[loc+1] != '*') && (input[loc+1] != '/')){
//              return expi;
//          }else{
//              if ((op.find('+') == op.end()) && (op.find('-') == op.end())){
//                  return expi;
//              }else{
//                  return '('+expi+')';
//              }
//          }
//        }else{
//            char temp = input[loc];
//            expi=expi+temp;
//            if(support.find(temp) != support.end()){
//                op.insert(temp);
//            }
//        }
//        loc++;
//        if(loc >= size){
//            break;
//        }
//    }
//
//    return expi;
//}
