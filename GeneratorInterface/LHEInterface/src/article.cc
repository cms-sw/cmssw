// MCDB API: article.cpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#include "GeneratorInterface/LHEInterface/interface/mcdb.h"
#include "GeneratorInterface/LHEInterface/src/macro.h"

namespace mcdb {

using std::string;
using std::vector;    


Article::Article(): id_(0) { }
Article::~Article() { }

FUNC_SETGET(int&, Article, id)
FUNC_SETGET(string&, Article, title)
FUNC_SETGET(string&, Article, abstract)
FUNC_SETGET(string&, Article, comments)
FUNC_SETGET(string&, Article, experiment)
FUNC_SETGET(string&, Article, group)
FUNC_SETGET(vector<Author>&, Article, authors)
FUNC_GET(const string, Article, postDate)
FUNC_SETGET(Process&, Article, process)
FUNC_SETGET(vector<Subprocess>&, Article, subprocesses)
FUNC_SETGET(Generator&, Article, generator)
FUNC_SETGET(Model&, Article, model)
FUNC_SETGET(vector<Cut>&, Article, cuts)
FUNC_SETGET(vector<File>&, Article, files)
FUNC_SETGET(vector<string>&, Article, relatedPapers)


Author::Author() { }
Author::~Author() { }

FUNC_SETGET(string&, Author, firstName)
FUNC_SETGET(string&, Author, lastName)
FUNC_SETGET(string&, Author, email)
FUNC_SETGET(string&, Author, experiment)
FUNC_SETGET(string&, Author, expGroup)
FUNC_SETGET(string&, Author, organization)


Model::Model() { }
Model::~Model() { }

Model::ModelParameter::ModelParameter() { }
Model::ModelParameter::~ModelParameter() { }

FUNC_SETGET(string&,Model,name)
FUNC_SETGET(string&,Model,description)
FUNC_SETGET(vector<Model::ModelParameter>&,Model,parameters)
FUNC_SETGET(string&,Model::ModelParameter,name)
FUNC_SETGET(string&,Model::ModelParameter,value)


Generator::Generator() { }
Generator::~Generator() { }

FUNC_SETGET(string&, Generator, name)
FUNC_SETGET(string&, Generator, version)
FUNC_SETGET(string&, Generator, homepage)


Cut::Cut(): logic_(include_region) { }
Cut::~Cut() { }

FUNC_SETGET(string&, Cut, object)
FUNC_SETGET(string&, Cut, minValue)
FUNC_SETGET(string&, Cut, maxValue)
FUNC_SETGET(CutLogic&, Cut, logic)


Process::Process() { }
Process::~Process() { }

FUNC_SETGET(string&,Process,initialState)
FUNC_SETGET(string&,Process,finalState)
FUNC_SETGET(string&,Process,factScale)
FUNC_SETGET(string&,Process,renormScale)
FUNC_SETGET(string&,Process,pdf)


Subprocess::Subprocess() { }
Subprocess::~Subprocess() { }

FUNC_SETGET(string&, Subprocess, notation)
FUNC_SETGET(float&, Subprocess, crossSectionPb)
FUNC_SETGET(float&, Subprocess, csErrorPlusPb)
FUNC_SETGET(float&, Subprocess, csErrorMinusPb)

} //namespace mcdb
