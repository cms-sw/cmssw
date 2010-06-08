class dibin{
  
 public:
  dibin():i(0),j(0),k(0),l(0) {}
  
  dibin(int the_n,int the_m, char tk)
      {
	if(tk == 'I')
	  { i= the_n; j= the_m; k= 0; l= 0; }
	else if(tk == 'T')
	  { i = 0; j = 0; k = the_n; l = the_m; }
      }
    

    dibin (int the_i, int the_j, int the_k, int the_l)
      { i = the_i; j=the_j; k=the_k; l=the_l; }
    

    dibin (const dibin& d)
      { i = d.i; j = d.j; k = d.k; l = d.l; }
    
    int GetOuterIDKey()const {return i;}
    int GetInnerIDKey()const {return j;}    
    int GetOuterTrKey()const {return k;}
    int GetInnerTrKey()const {return l;}

    bool operator<(const dibin& dbn) const {
      return (i<dbn.i || ((i==dbn.i)&&(j<dbn.j))||
	      ((i==dbn.i)&&(dbn.j==j)&&(k<dbn.k))||
	      ((i==dbn.i)&&(dbn.j==j)&&(k==dbn.k)&&(l<dbn.l) ));
    }
    
    bool operator==(const dibin& dbn)const { 
      return ((i==dbn.i)&&(j==dbn.j)&&(k==dbn.k)&& (l==dbn.l));
    }

    void print (int* ar) const {
      ar[0] = i;
      ar[1] = j;
      ar[2] = k;
      ar[3] = l;
    }
    
 private:
    int i;
    int j;
    int k;
    int l;
};


