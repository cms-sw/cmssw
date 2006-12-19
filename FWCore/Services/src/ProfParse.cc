
#include <exception>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <stdexcept>
#include <set>
#include <vector>
#include <deque>
#include <map>
#include <cstdlib>
#include <cerrno>

#include <dlfcn.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if 0
extern "C" {
  // options:
#define DMGL_AUTO    (1 << 8)
  extern char* cplus_demangle(const char *mangled, int options);
}
#endif

using namespace std;

typedef vector<void*> VoidVec;
typedef vector<unsigned int> ULVec;
typedef map<unsigned int,int> EdgeMap;

struct Sym
{
  explicit Sym(int id):id_(id),addr_() { }
  Sym():id_(),addr_() { }

  int id_;
  unsigned int addr_;
  std::string name_;
  int size_;

  static int next_id_;

  bool operator<(unsigned int b) const
    { return addr_ < b; }
  bool operator<(const Sym& b) const
    { return addr_ < b.addr_; }
};

inline bool operator<(unsigned int a, const Sym& b)
{ return a < b.addr_; }

std::ostream& operator<<(std::ostream& ost,const Sym& s)
{
  ost << s.id_ << " " << s.addr_ << " " << s.name_ << " " << s.size_;
  return ost;
}

int Sym::next_id_ = 1000000;

// ------------------- Vertex Tracker class ----------------
struct VertexTracker
{
  VertexTracker():
    name_(),addr_(),id_(),total_as_leaf_(),total_seen_(),in_path_(),size_(),
    percent_leaf_(),percent_path_()
  { init(); } 
  explicit VertexTracker(unsigned int id):
    name_(),addr_(),id_(id),total_as_leaf_(),total_seen_(),in_path_(),size_(),
    percent_leaf_(),percent_path_()

  { init(); } 
  VertexTracker(unsigned int addr, const string& name):
    name_(name),addr_(addr),id_(),total_as_leaf_(),total_seen_(),
    in_path_(),size_(),percent_leaf_(),percent_path_()
  { init(); }

  explicit VertexTracker(const Sym& e);

  bool operator<(const VertexTracker& a) const { return addr_<a.addr_; }
  bool operator<(unsigned int id) const { return id_<id; }

  void incLeaf() const { ++total_as_leaf_; }
  void incTotal() const { ++total_seen_; }
  void incPath(int by) const { in_path_+=by; }
  void setID() const { id_=next_id_++; }
  void fillHist(unsigned int addr_within_func) const;
  void init();

  string name_;
  unsigned int addr_;
  mutable unsigned int id_;
  mutable unsigned int total_as_leaf_;
  mutable unsigned int total_seen_;
  mutable unsigned int in_path_;
  mutable EdgeMap edges_;
  static const int wheresize = 10;
  mutable int where_[wheresize+1];
  mutable int size_;
  mutable float percent_leaf_;
  mutable float percent_path_;

  static unsigned int next_id_;

};

void VertexTracker::fillHist(unsigned int value) const
{
  if(value<addr_ || size_==0) return;

  unsigned int offset = value - addr_;
  unsigned int index = (offset * wheresize) / size_;
  /*
  cout << "index:" << index 
       << " size:" << size_
       << " offset:" << offset
       << " wheresize:" << wheresize
       << endl;
  */
  if(index > (unsigned int)wheresize) return;
  where_[index]+=1;
}

VertexTracker::VertexTracker(const Sym& e):
  name_(e.name_),addr_(e.addr_),id_(),
  total_as_leaf_(),total_seen_(),in_path_(),size_(e.size_),
  percent_leaf_(),percent_path_()
{
  init();
}

void VertexTracker::init()
{
  percent_leaf_=0.0;
  percent_path_=0.0;
  for(int i=0;i<=wheresize;++i) where_[i]=0;
}

unsigned int VertexTracker::next_id_ = 0;

ostream& operator<<(ostream& ost, const VertexTracker& a)
{
  static int empty_count = 1;
  string n = a.name_;
  // this is a bad place for this code
  if(n.empty())
    {
      ostringstream ostr;
      ostr << "EmptyName-" << empty_count;
      ++empty_count;
      n = ostr.str();
    }

  ost << a.id_ << " "
      << (void*)a.addr_ << " "
      << n << " "
      << a.total_as_leaf_ << " "
      << a.total_seen_ << " "
      << a.in_path_ << " "
      << a.percent_leaf_ << " "
      << a.percent_path_ << " ";

  for(int i=0;i<a.wheresize;++i)
    {
      ost << a.where_[i] << " ";
    }

  return ost;
}

// ----------------- Path tracker class ---------------------

struct PathTracker
{
  PathTracker(): id_(),total_() { }

  mutable unsigned int id_;
  mutable unsigned int total_;
  ULVec tree_;

  bool operator<(const PathTracker& a) const;
  void setID() const { id_=next_id_++; }
  void incTotal() const { ++total_; }

  static unsigned int next_id_;
};

unsigned int PathTracker::next_id_ = 0;

bool PathTracker::operator<(const PathTracker& a) const
{
  return tree_ < a.tree_;
}

ostream& operator<<(ostream& ost, const PathTracker& a)
{
  ost << a.id_ << " " << a.total_ << " ";
  ULVec::const_iterator i(a.tree_.begin()),e(a.tree_.end());
  while(i!=e) { ost << (unsigned int)*i << " "; ++i; }
  return ost;
}

// ------------------- utilities --------------------------

void verifyFile(ostream& ost, const string& name)
{
  if(!ost)
    {
      cerr << "cannot open output file " << name << endl;
      throw runtime_error("failed to open output file");
    }
}

// ---------------- some definitions ------------------

typedef set<VertexTracker> VertexSet;
typedef set<PathTracker> PathSet;
typedef vector<VertexSet::const_iterator> Viter;
typedef vector<PathSet::const_iterator> Piter;

// ------------- more utilities ----------------

static bool symSort(const VertexSet::const_iterator& a,
		    const VertexSet::const_iterator& b)
{
  return a->total_as_leaf_ < b->total_as_leaf_;
}

static bool idSort(const VertexSet::const_iterator& a,
		    const VertexSet::const_iterator& b)
{
  return a->id_ < b->id_;
}

static bool idComp(unsigned int id,
		    const VertexSet::const_iterator& b)
{
  return id < b->id_;
}

static bool pathSort(const PathSet::const_iterator& a,
		    const PathSet::const_iterator& b)
{
  return a->total_ < b->total_;
}

// ------------------ main routine --------------------

class Reader
{
 public:
  explicit Reader(int fd):fd_(fd) { }
  bool nextSample(VoidVec& vv);
 private:
  int fd_;
};

bool Reader::nextSample(VoidVec& vv)
{
  unsigned int cnt;
  int sz = read(fd_,&cnt,sizeof(unsigned int));

  if(sz<0)
    {
      perror("Reader::nextSample: read count");
      cerr << "could not read next sample from profile data\n";
      return false;
    }
  if(sz==0) return false;
  if((unsigned)sz<sizeof(unsigned int))
    {
      cerr << "Reader::nextSample: "
	   << "could not read the correct amount of profile data\n";
      return false;
    }
  if(cnt>1000)
    {
      cerr << "Reader::nextSample: stack length is nonsense " << cnt << "\n";
      return false;
    }
  
  vv.resize(cnt);
  void** pos = &vv[0];
  int byte_cnt = cnt*sizeof(void*);

  while((sz=read(fd_,pos,byte_cnt))<byte_cnt)
    {
      if(sz<0)
	{
	  perror("Reader::nextSample: read stack");
	  cerr << "could not read stack\n";
	  return false;
	}
      byte_cnt-=sz;
      pos+=sz;
    }
  return true;
}

void writeProfileData(int fd, const std::string& prefix)
{
  string output_tree(prefix+"_paths");
  string output_names(prefix+"_names");
  string output_totals(prefix+"_totals");

  ofstream nost(output_names.c_str());
  ofstream tost(output_tree.c_str());
  ofstream sost(output_totals.c_str());

  verifyFile(nost,output_names);
  verifyFile(tost,output_tree);
  verifyFile(sost,output_totals);

  VertexSet symset;
  PathSet pathset;
  pair<VertexSet::iterator,bool> irc,prev_irc;
  pair<PathSet::iterator,bool> prc;

  VoidVec v;
  int len=0;
  int total=0;
  int unknown_count=1;
  Sym last_none_entry;
  Sym last_good_entry;
  Reader r(fd);
  string unk("unknown_name");

  while(r.nextSample(v)==true)
    {
      PathTracker ptrack;
      ++total;
      len = v.size();
      if(len==0) continue; // should never happen!
      VoidVec::reverse_iterator c(v.rbegin()),e(v.rend());
      bool first_pass=true;

      while(c!=e)
	{
	  unsigned int value = reinterpret_cast<unsigned int>(*c);

	  const Sym* entry = 0;
	  Dl_info look;

#if 0  
	  if(dladdr((void*)value,&look)!=0)
	    {
	      cerr << look.dli_fname 
		   << ":" << (look.dli_saddr ? look.dli_sname : "?")
		   << ":" << look.dli_saddr
		   << "\n--------\n";      
	    }
#endif
	  if(dladdr((void*)value,&look)!=0)
	    {
	      string nam = (look.dli_saddr ? look.dli_sname : "unknown_name");

	      last_good_entry.id_ = 0;
	      last_good_entry.name_ = nam;
	      last_good_entry.addr_ = (unsigned int)look.dli_saddr;
	      last_good_entry.size_ = 0;
	      entry = &last_good_entry;
	    }
	    
	  if(entry==0)
	    {
	      cerr << "sample " << total
		   << ": could not find function for address " << *c
		   << endl;
	      ostringstream uost;
	      uost << "unknown" << unknown_count;
	      ++unknown_count;
	      entry = &last_none_entry;
	      last_none_entry.id_ = Sym::next_id_++;
	      last_none_entry.name_ = uost.str();
	      last_none_entry.addr_ = value;
	    }

	  irc = symset.insert(VertexTracker(*entry));
	  if(irc.second)
	    {
	      irc.first->setID();
	      //cout << "new node: " << *irc.first << endl;
	    }
	  irc.first->incTotal();
	  irc.first->fillHist(value);
	  ptrack.tree_.push_back(irc.first->id_);
	  //cout << "added to tree: " << irc.first->id_ << endl;

	  if(!first_pass) ++prev_irc.first->edges_[irc.first->id_];
	  else first_pass=false;

	  prev_irc = irc;
	  ++c;
	}

      irc.first->incLeaf();
      prc = pathset.insert(ptrack);
      if(prc.second)
	{
	  prc.first->setID();
	}
      //cout << "new path \n" << *prc.first << endl;
      prc.first->incTotal();
    }  

  // ------------------ copy the vertices for sorting and searching ------

  int setsize = symset.size();
  int edgesize = 0;
  Viter vsyms;
  vsyms.reserve(setsize);

  //cout << "------ symset -----" << endl;
  VertexSet::const_iterator isym(symset.begin()),esym(symset.end());
  while(isym!=esym)
    {
      //cout << "     " << *isym << endl;
      vsyms.push_back(isym);
      ++isym;
    }

  // ------ calculate samples for parents and percentages in vertices ------

  sort(vsyms.begin(),vsyms.end(),idSort);
  //Viter::iterator Vib(vsyms.begin()),Vie(vsyms.end());
  //cout << "sorted table --------------" << endl;
  //while(Vib!=Vie) { cout << "    " << *(*Vib) << endl; ++Vib; }

  PathSet::const_iterator pat_it_beg(pathset.begin()),
    pat_it_end(pathset.end());

  while(pat_it_beg!=pat_it_end)
    {
      // get set of unique IDs from the path
      ULVec pathcopy(pat_it_beg->tree_);
      sort(pathcopy.begin(),pathcopy.end());
      ULVec::iterator iter = unique(pathcopy.begin(),pathcopy.end());
      ULVec::iterator cop_beg(pathcopy.begin());
      //cout << "length of unique = " << distance(cop_beg,iter) << endl;
      while(cop_beg!=iter)
	{
	  //cout << "  entry " << *cop_beg << endl;
	  Viter::iterator sym_iter = upper_bound(vsyms.begin(),vsyms.end(),
						 *cop_beg,idComp);
	  if(sym_iter==vsyms.begin())
	    {
	      cerr << "found a missing sym entry for address " << *cop_beg
		   << endl;
	    }
	  else
	    {
	      --sym_iter;
	      //cout << " symiter " << *(*sym_iter) << endl;
	      (*sym_iter)->incPath(pat_it_beg->total_);
	    }
	  ++cop_beg;
	}

      ++pat_it_beg;
    }

  VertexSet::iterator ver_iter(symset.begin()),ver_iter_end(symset.end());
  while(ver_iter!=ver_iter_end)
    {
      ver_iter->percent_leaf_ = (float)ver_iter->total_as_leaf_ / (float)total;
      ver_iter->percent_path_ = (float)ver_iter->in_path_ / (float)total;
      ++ver_iter;
    }

  // -------------- write out the vertices ----------------

  // eost << "digraph prof {\n";

  sort(vsyms.begin(),vsyms.end(),symSort);
  Viter::reverse_iterator vvi(vsyms.rbegin()),vve(vsyms.rend());
  while(vvi!=vve)
    {
      EdgeMap::const_iterator id((*vvi)->edges_.begin()),
	ed((*vvi)->edges_.end());
      while(id!=ed)
	{
	  //eost << "\t" << (*vvi)->id_ << " -> " << id->first 
	  //     << " [label=\"" << id->second << "\"];\n";
	  //eost << id->second << " " << (*vvi)->id_ << " " << id->first <<"\n";
	  ++edgesize;
	  ++id;
	}

      nost << *(*vvi) << "\n";
      ++vvi;
    }

  // eost << "}" << endl;

  // --------------- write out the paths ------------------ 

  int pathsize = pathset.size();
  Piter vpaths;
  vpaths.reserve(pathsize);

  PathSet::const_iterator ipath(pathset.begin()),epath(pathset.end());
  while(ipath!=epath)
    {
      vpaths.push_back(ipath);
      ++ipath;
    }

  sort(vpaths.begin(),vpaths.end(),pathSort);
  Piter::reverse_iterator ppi(vpaths.rbegin()),ppe(vpaths.rend());
  while(ppi!=ppe)
    {
      tost << *(*ppi) << "\n";
      ++ppi;
    }

  // ------------ totals --------------------
  sost << "total_samples " << total << "\n"
       << "total_functions " << setsize << "\n"
       << "total_paths " << pathsize << "\n"
       << "total_edges " << edgesize << endl;

}

