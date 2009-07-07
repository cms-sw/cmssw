//$Id: SprTrainedRBF.cc,v 1.2 2007/09/21 22:32:10 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTrainedRBF.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprTransformation.hh"

#include <map>
#include <utility>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cassert>

using namespace std;


bool SprTrainedRBF::readNet(const char* netfile)
{
  // open file
  string fname = netfile;
  ifstream file(fname.c_str());
  if( !file ) {
    cerr << "Unable to open file " << fname.c_str() << endl;
    return false;
  }
 
  // read junk on top of the file
  string line;
  unsigned nline = 1;
  int nempty = 5;
  for( int i=0;i<nempty;i++ ) {
    if( !getline(file,line) ) {
      cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
      return false;
    }
    nline++;
  }

  // read number of nodes and links
  unsigned nnodes(0), nlinks(0);
  if( !getline(file,line) ) {
    cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
    return false;
  }
  else {
    nline++;
    line.erase( 0, line.find_first_of(':')+1 );
    istringstream ist(line);
    ist >> nnodes;
    if( nnodes == 0 ) {
      cerr << "No nodes found in " << fname.c_str() << endl;
      return false;
    }
  }
  if( !getline(file,line) ) {
    cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
    return false;
  }
  else {
    nline++;
    line.erase( 0, line.find_first_of(':')+1 );
    istringstream ist(line);
    ist >> nlinks;
    if( nlinks == 0 ) {
      cerr << "No links found in " << fname.c_str() << endl;
      return false;
    }
  }

  // more empty lines
  nempty = 4;
  for( int i=0;i<nempty;i++ ) {
    if( !getline(file,line) ) {
      cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
      return false;
    }
    nline++;
  }

  // read learning and update function
  if( !getline(file,line) ) {
    cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
    return false;
  }
  else {
    nline++;
    line.erase( 0, line.find_first_of(':')+1 );
    line.erase( line.find_last_not_of(' ')+1 );
    line.erase( 0, line.find_first_not_of(' ') );
    if( line != "RadialBasisLearning" ) {
      cerr << "Learning function is not RadialBasisLearning!!!" << endl;
      return false;
    }
  }
  if( !getline(file,line) ) {
    cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
    return false;
  }
  else {
    nline++;
    line.erase( 0, line.find_first_of(':')+1 );
    line.erase( line.find_last_not_of(' ')+1 );
    line.erase( 0, line.find_first_not_of(' ') );
    if( line != "Topological_Order" ) {
      cerr << "Update function is not Topological_Order!!!" << endl;
      return false;
    }
  }

  // more empty lines
  nempty = 6;
  for( int i=0;i<nempty;i++ ) {
    if( !getline(file,line) ) {
      cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
      return false;
    }
    nline++;
  }

  // read activation and output functions (6th and 7th fields)
  SprNNDefs::ActFun baseAct;
  SprNNDefs::OutFun baseOut;
  if( !getline(file,line) ) {
    cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
    return false;
  }
  else {
    nline++;
    for( unsigned int i=0;i<5;i++ )
      line.erase( 0, line.find_first_of('|')+1 );
    string act = line.substr( 0, line.find_first_of('|') );
    string out = line.substr( line.find_first_of('|')+1 );
    act.erase( 0, act.find_first_not_of(' ') );
    act.erase( act.find_last_not_of(' ')+1 );
    out.erase( 0, out.find_first_not_of(' ') );
    out.erase( out.find_last_not_of(' ')+1 );
    if(      act == "Act_Logistic" )
      baseAct = SprNNDefs::LOGISTIC;
    else if( act=="Act_Identity" || act=="Act_IdentityPlusBias" )
      baseAct = SprNNDefs::ID;
    else {
      cerr << "Unknown activation function " << act.c_str() 
	   << " in " << fname.c_str() << endl;
      return false;
    }
    if( out == "Out_Identity" )
      baseOut = SprNNDefs::OUTID;
    else {
      cerr << "Unknown output function " << out.c_str() 
	   << " in " << fname.c_str() << endl;
      return false;
    }
  }

  // more empty lines
  nempty = 7;
  for( int i=0;i<nempty;i++ ) {
    if( !getline(file,line) ) {
      cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
      return false;
    }
    nline++;
  }

  // read units section
  for( unsigned int i=0;i<nnodes;i++ ) {
    //    cout << "Reading node " << (i+1) << endl;
    if( !getline(file,line) ) {
      cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
      return false;
    }
    else {
      nline++;
      // read index
      string piece = line.substr( 0, line.find_first_of('|') );
      line.erase( 0, line.find_first_of('|')+1 );
      istringstream stindex(piece);
      unsigned index(0);
      stindex >> index;
      assert( index != 0 );
      // read activation
      for( unsigned int j=0;j<2;j++ )
	line.erase( 0, line.find_first_of('|')+1 );
      piece = line.substr( 0, line.find_first_of('|') );
      line.erase( 0, line.find_first_of('|')+1 );
      istringstream stact(piece);
      double act = 0;
      stact >> act;
      // read bias
      piece = line.substr( 0, line.find_first_of('|') );
      line.erase( 0, line.find_first_of('|')+1 );
      istringstream stbias(piece);
      double bias = 0;
      stbias >> bias;
      // read node type
      piece = line.substr( 0, line.find_first_of('|') );
      piece.erase( 0, piece.find_first_not_of(' ') );
      piece.erase( piece.find_last_not_of(' ')+1 );
      line.erase( 0, line.find_first_of('|')+1 );
      SprNNDefs::NodeType type;
      if(      piece == "i" )
	type = SprNNDefs::INPUT;
      else if( piece == "h" )
	type = SprNNDefs::HIDDEN;
      else if( piece == "o" )
	type = SprNNDefs::OUTPUT;
      else {
	cerr << "Unknown node type " << piece.c_str() 
	     << " in " << fname.c_str() << endl;
	return false;
      }
      // read activation function
      line.erase( 0, line.find_first_of('|')+1 );
      piece = line.substr( 0, line.find_first_of('|') );
      piece.erase( 0, piece.find_first_not_of(' ') );
      piece.erase( piece.find_last_not_of(' ')+1 );
      line.erase( 0, line.find_first_of('|')+1 );
      SprNNDefs::ActFun actfun = baseAct;
      ActRBF actrbf = Gauss;
      if( type == SprNNDefs::HIDDEN ) {
	if(      piece == "Act_RBF_Gaussian" )
	  actrbf = Gauss;
	else if( piece == "Act_RBF_MultiQuadratic" )
	  actrbf = MultiQ;
	else if( piece == "Act_RBF_ThinPlateSpline" )
	  actrbf = ThinPlate;
	else {
	  cerr << "Unknown RBF activation function " << piece.c_str() 
	       << " in " << fname.c_str() << endl;
	  return false;
	}
      }
      else {// not a hidden node
	if( !piece.empty() ) {
	  if(      piece == "Act_Logistic" )
	    actfun = SprNNDefs::LOGISTIC;
	  else if( piece=="Act_Identity" || piece=="Act_IdentityPlusBias" )
	    actfun = SprNNDefs::ID;
	  else {
	    cerr << "Unknown activation function " << piece.c_str() 
		 << " in " << fname.c_str() << endl;
	    return false;
	  }
	}
      }
      // read output function
      piece = line.substr( 0, line.find_first_of('|') );
      piece.erase( 0, piece.find_first_not_of(' ') );
      piece.erase( piece.find_last_not_of(' ')+1 );
      line.erase( 0, line.find_first_of('|')+1 );
      SprNNDefs::OutFun outfun = baseOut;
      if( !piece.empty() ) {
	if( piece == "Out_Identity" )
	  outfun = SprNNDefs::OUTID;
	else {
	  cerr << "Unknown output function " << piece.c_str() 
	       << " in " << fname.c_str() << endl;
	  return false;
	}
      }
      // make a node
      Node* node = new Node();
      node->index_ = index;
      node->type_ = type;
      node->actFun_ = actfun;
      node->outFun_ = outfun;
      node->actRBF_ = actrbf;
      node->bias_ = bias;
      node->act_ = act;
      nodes_.push_back(node);
      assert( index == nodes_.size() ); 
    }
  }

  // more empty lines
  nempty = 7;
  for( int i=0;i<nempty;i++ ) {
    if( !getline(file,line) ) {
      cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
      return false;
    }
    nline++;
  }

  // read links
  unsigned readlinks = 0;
  while( readlinks < nlinks ) {
    //    cout << "Reading link " << (readlinks+1) << endl;
    if( !getline(file,line) ) {
      cerr << "Error on line " << nline << " in " << fname.c_str() << endl;
      return false;
    }
    else {
      nline++;
      // read index
      string piece = line.substr( 0, line.find_first_of('|') );
      line.erase( 0, line.find_first_of('|')+1 );
      istringstream stindex(piece);
      unsigned index(0);
      stindex >> index;
      assert( index != 0 );
      // read sources and weights
      line.erase( 0, line.find_first_of('|')+1 );
      piece = line.substr( 0, line.find_first_of('|') );
      piece.erase( piece.find_last_not_of(' ')+1 );
      while( !piece.empty() ) {
	string srcwt;
	if( piece.find(',') != string::npos ) {
	  srcwt = piece.substr( 0, piece.find_first_of(',') );
	  piece.erase( 0, piece.find_first_of(',')+1 );
	  piece.erase( piece.find_last_not_of(' ')+1 );
	  if( piece.empty() ) {
	    if( !getline(file,piece) ) {
	      cerr << "Unable to read line " << nline 
		   << " in " << fname.c_str() << endl;
	      return false;
	    }
	    else
	      nline++;
	  }
	}
	else {
	  srcwt = piece;
	  piece.clear();
	}
	srcwt.erase( 0, srcwt.find_first_not_of(' ') );
	srcwt.erase( srcwt.find_last_not_of(' ')+1 );
	unsigned src = atoi(srcwt.substr(0,srcwt.find_first_of(':')).c_str());
	double wt = atof(srcwt.substr(srcwt.find_first_of(':')+1).c_str());
	assert( src != 0 );
	// insert link
	Link* link = new Link();
	link->weight_ = wt;
	Node* target = nodes_[index-1];
	Node* source = nodes_[src-1];
	target->incoming_.push_back(link);
	source->outgoing_.push_back(link);
	link->source_ = source;
	link->target_ = target;
	links_.push_back(link);
	readlinks++;
      }
    }
  }

  // success
  return true;
}


void SprTrainedRBF::printNet(std::ostream& os) const
{
  os << "Nodes of RBF network:" << endl;
  for( unsigned int i=0;i<nodes_.size();i++ ) {
    const Node* node = nodes_[i];
    os << node->index_ 
       << " Type " << int(node->type_)
       << " ActFun " << int(node->actFun_)
       << " ActRBF " << int(node->actRBF_)
       << " OutFun " << int(node->outFun_)
       << " activation " << node->act_
       << " bias " << node->bias_
       << endl;
  }
  os << "Links of RBF network:" << endl;
  for( unsigned int i=0;i<links_.size();i++ ) {
    const Link* link = links_[i];
    os << " Source " << link->source_->index_
       << " Target " << link->target_->index_
       << " weight " << link->weight_
       << endl;
  }
}


double SprTrainedRBF::response(const std::vector<double>& v) const
{
  // loop over hidden nodes and compute RBF values
  map<unsigned,double> hidden;// RBF values at hidden nodes
  for( unsigned int i=0;i<nodes_.size();i++ ) {
    const Node* node = nodes_[i];
    if( node->type_ == SprNNDefs::HIDDEN ) {
      /*
      cout << node->index_ << endl;
      cout << node->incoming_.size() << " " << v.size() << endl;
      */
      assert( node->incoming_.size() == v.size() );
      double r2 = 0;// r squared
      for( unsigned int j=0;j<node->incoming_.size();j++ ) {
	const Link* link = node->incoming_[j];
	assert( link->source_->type_ == SprNNDefs::INPUT );
	double x_t = v[link->source_->index_-1] - link->weight_;
	r2 += x_t * x_t;
      }
      hidden.insert(pair<const unsigned,
		    double>(node->index_,
			    this->rbf(r2,node->bias_,node->actRBF_)));
    }
  }

  // loop over output nodes and sum linear contributions from input nodes
  // and RBF contributions from hidden nodes
  vector<double> output;
  for( unsigned int i=0;i<nodes_.size();i++ ) {
    const Node* node = nodes_[i];
    if( node->type_ == SprNNDefs::OUTPUT ) {
      output.push_back(0);
      int imax = output.size()-1;
      for( unsigned int j=0;j<node->incoming_.size();j++ ) {
	const Link* link = node->incoming_[j];
	if(      link->source_->type_ == SprNNDefs::INPUT )
	  output[imax] += v[link->source_->index_-1] * (link->weight_);
	else if( link->source_->type_ == SprNNDefs::HIDDEN )
	  output[imax] += hidden[link->source_->index_] * (link->weight_);
      }
      output[imax] = this->act(output[imax],node->bias_,node->actFun_);
    }
  }
  assert( !output.empty() );

  // return value of the first output node
  return output[0];
}



void SprTrainedRBF::destroy()
{
  for( unsigned int i=0;i<nodes_.size();i++ )
    delete nodes_[i];
  for( unsigned int i=0;i<links_.size();i++ )
    delete links_[i];
}


void SprTrainedRBF::correspondence(const SprTrainedRBF& other)
{
  // links
  map<Link*,Link*> ltol;
  for( unsigned int i=0;i<other.links_.size();i++ ) {
    Link* old = other.links_[i];
    Link* link = new Link(*old);
    links_.push_back(link);
    ltol.insert(pair<Link* const,Link*>(old,link));
  }

  // nodes
  map<Node*,Node*> nton;
  for( unsigned int i=0;i<other.nodes_.size();i++ ) {
    Node* old = other.nodes_[i];
    Node* node = new Node(*old);
    nodes_.push_back(node);
    nton.insert(pair<Node* const,Node*>(old,node));
  }

  // adjust links
  for( unsigned int i=0;i<nodes_.size();i++ ) {
    Node* node = nodes_[i];
    for( unsigned int j=0;j<(node->incoming_).size();j++ )
      (node->incoming_)[j] = ltol[(node->incoming_)[j]];
    for( unsigned int j=0;j<(node->outgoing_).size();j++ )
      (node->outgoing_)[j] = ltol[(node->outgoing_)[j]];
  }

  // adjust nodes
  for( unsigned int i=0;i<links_.size();i++ ) {
    Link* link = links_[i];
    link->source_ = nton[link->source_];
    link->target_ = nton[link->target_];
  }
}


double SprTrainedRBF::rbf(double r2, double p, ActRBF act) const
{
  switch( act )
    {
    case Gauss :
      return exp(-r2*p);
      break;
    case MultiQ :
      return ( (r2+p)>0 ? sqrt(r2+p) : 0 );
      break;
    case ThinPlate :
      return ( (r2>0&&p>0) ? p*p*r2*log(p*sqrt(r2)) : 0 );
      break;
    }
  return 0;
}


double SprTrainedRBF::act(double x, double p, SprNNDefs::ActFun act) const
{
  switch( act )
    {
    case SprNNDefs::ID :
      return (x+p);
      break;
    case SprNNDefs::LOGISTIC :
      return SprTransformation::logit(x+p);
      break;
    }
  return 0;
}
