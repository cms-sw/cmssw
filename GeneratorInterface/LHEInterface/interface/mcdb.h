// mcdb.hpp
// MCDB API public include header
// part of LCG MCDB project, http://mcdb.cern.ch
// Monte Carlo Data Base
// Sergey Belov <Sergey.Belov@cern.ch>, 2006-2007

#ifndef MCDB_HPP_
#define MCDB_HPP_ 1

#include <string>
#include <vector>

namespace mcdb
{

using std::string;
using std::vector;

// Possible file types in MCDB:
// 'data' - sample file, 'meta' - file containing some metadata,
// 'other' - something else
enum FileType {any=0, data, metadata, generator_config, generator_binary};

// Logic of applied cut restriction
enum CutLogic {include_region, exclude_region};


class Article;
class File;
class Author;
class Cut;
class Generator;
class Model;
class Process;
class Subprocess;

// MCDB class provides access to MCDB itself
class MCDB{
 public:
    MCDB();
    MCDB(const string& baseUrl);
    ~MCDB();
    const Article getArticle(int id);
    const Article getArticle(const string& Uri);   
    const vector<File> getFiles(int articleId);
    const vector<File> getFiles(const string& articleXmlUri);
    string& serverBaseUrl();
    string& serverBaseUrl(const string&);
    bool& hepmlProcessNs();
    bool& hepmlProcessNs(const bool&);
    bool& hepmlReportErrors();
    bool& hepmlReportErrors(const bool&);
    int   errorCode();
 private:
    string serverBaseUrl_;
    bool hepmlProcessNs_;
    bool hepmlReportErrors_;
    int  errorCode_;
};


// MC generator information
class Generator{
 public:
    Generator();
    ~Generator();
    string& name();
    string& name(const string&);
    string& version();
    string& version(const string&);
    string& homepage();
    string& homepage(const string&);
 private:    
    string name_;
    string version_;
    string homepage_;
}; // class Generator


// Physical model
class Model{
 public:
    Model();
    ~Model();
    class ModelParameter;
    string& name();
    string& name(const string&);
    string& description();
    string& description(const string&);
    vector<ModelParameter>& parameters();
    vector<ModelParameter>& parameters(const vector<ModelParameter>&);
    class ModelParameter
    {
     public:
        ModelParameter();
        ~ModelParameter();
        string& name();
        string& name(const string&);
        string& value();
        string& value(const string&);
     private:
        string name_;
        string value_;
    };
 private:    
    string name_;
    string description_;
    vector<ModelParameter> parameters_;
}; // class Model


// Physical process description
class Process{
 public:
    Process();
    ~Process();
    string& initialState();
    string& initialState(const string&); 
    string& finalState();
    string& finalState(const string&);
    string& factScale();
    string& factScale(const string&);
    string& renormScale();
    string& renormScale(const string&);
    string& pdf();
    string& pdf(const string&);
 private:
    string initialState_;
    string finalState_;
    string factScale_;
    string renormScale_;
    string pdf_;
}; // class Process


class Subprocess{
 public:
    Subprocess();
    ~Subprocess();
    string& notation();
    string& notation(const string&);
    float& crossSectionPb();
    float& crossSectionPb(const float&);
    float& csErrorPlusPb();
    float& csErrorPlusPb(const float&); 
    float& csErrorMinusPb();
    float& csErrorMinusPb(const float&);
 private:
    string notation_;
    float crossSectionPb_;
    float csErrorPlusPb_;
    float csErrorMinusPb_;
};

// Information of samples' author
class Author{    
 public:
    Author();
    ~Author();
    string& firstName();
    string& firstName(const string&);
    string& lastName();
    string& lastName(const string&);
    string& email();
    string& email(const string&);
    string& experiment();
    string& experiment(const string&);
    string& expGroup();
    string& expGroup(const string&);
    string& organization();
    string& organization(const string&);
 private:
    string firstName_;
    string lastName_;
    string email_;
    string experiment_;
    string expGroup_;
    string organization_;
}; // class Author


// Applied cut
class Cut{
 public:
    Cut();
    ~Cut();
    string& object();
    string& object(const string&);
    string& minValue();
    string& minValue(const string&);
    string& maxValue();
    string& maxValue(const string&);
    CutLogic& logic();
    CutLogic& logic(const CutLogic&);
 private:
    string object_;
    string minValue_;
    string maxValue_;
    CutLogic logic_;
}; // class Cut


// Event sample/metadata file description
class File{
 public:
    File();
    ~File();
    FileType& type();
    FileType& type(const FileType&);
    int& id();
    int& id(const int&);
    int& eventsNumber();
    int& eventsNumber(const int&);
    unsigned long& size();
    unsigned long& size(const unsigned long&);
    string& checksum();
    string& checksum(const string&);
    float& crossSectionPb();
    float& crossSectionPb(const float&);
    float& csErrorPlusPb();
    float& csErrorPlusPb(const float&);
    float& csErrorMinusPb();
    float& csErrorMinusPb(const float&);
    string& comments();
    string& comments(const string&);
    vector<string>& paths();
    vector<string>& paths(const vector<string>&);
    vector<string> findPaths(const string& substr);
 private:    
    int eventsNumber_;
    float crossSectionPb_;
    float csErrorPlusPb_;
    float csErrorMinusPb_;
    unsigned long size_;
    string checksum_;
    string comments_;
    FileType type_;
    int id_;
    vector<string> paths_;
}; // class File


// Description for a set of event samples 
class Article{
 public:
    Article();
    ~Article();
    int& id();
    int& id(const int&);
    string& title();
    string& title(const string&);
    string& abstract();
    string& abstract(const string&);
    string& comments();
    string& comments(const string&);
    string& experiment();
    string& experiment(const string&);
    string& group();
    string& group(const string&);
    vector<Author>& authors();
    vector<Author>& authors(const vector<Author>&);
    const string postDate();
    Process& process();
    Process& process(const Process&);
    vector<Subprocess>& subprocesses();
    vector<Subprocess>& subprocesses(const vector<Subprocess>&);
    Generator& generator();
    Generator& generator(const Generator&); 
    Model& model();
    Model& model(const Model&);
    vector<Cut>& cuts();
    vector<Cut>& cuts(const vector<Cut>&);
    vector<File>& files();
    vector<File>& files(const vector<File>&);
    vector<string>& relatedPapers();
    vector<string>& relatedPapers(const vector<string>&);
 private:
    string title_;
    string abstract_;
    string comments_;
    string experiment_;
    string group_;
    string documentId_;
    string databaseId_;
    string postDate_;
    int id_;
    vector<File> files_;
    vector<string> relatedPapers_;
    Generator generator_; 
    Model model_;
    Process process_;
    vector<Subprocess> subprocesses_;
    vector<Cut> cuts_;
    vector<Author> authors_;
}; // Class Article


} // namespace mcdb

#endif /*MCDB_HPP_*/
