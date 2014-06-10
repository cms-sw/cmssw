double MyAnalysis (TH1D *h1, TH1D *h2) 
{
  const double nsigma=3;

  int naccepted=0;
  int nrejected=0;
  double result=0.0;

  double int1=h1->Integral();
  double int2=h2->Integral();

  if (int1==0) int1=1;
  if (int2==0) int2=1;


  // Thanks to T.P.!
  // ROOT enumerates bins from 1 to N, and not from 0 to N-1!
  // 0-th bin is underflow bin
  for (int i=1; i<=h1->GetNbinsX();i++) {

    double n1 = h1->GetBinContent(i);
    double n2 = h2->GetBinContent(i);
    
    double i1= n1/int1;
    double i2= n2/int2;
    

    // minimum of two bins:
    double n_min= (n1>n2) ? n2: n1;
    
    if ( n_min > 0) {


      // normalized division of bins:
      double nd=i2/i1;

      // sigma and acceptance strip
      double sigma = 1.0/sqrt(n_min);
      double accept=nsigma*sigma;

          
      if ( fabs(1.0-nd) <= accept ) {
	// inside acceptance strip!
	    
	naccepted++;
      } else {

	if (n1<n2) { // we must decrease n2 by 3sigma
	  double n_corr = n2 - accept*n2;
	  if (n_corr>n1) { // but not too much!
	    n2=n_corr;
	    result+=fabs(1.0-((n1/i1) / (n2/i2) )) / sigma /int1;
	  }
	      
	} else { // we must decrease n1 by 3sigma
	  double n_corr = n1- accept*n1;
	  if (n_corr>n2) { // but not too much!
	    n1=n_corr;
	    result+=fabs(1.0-((n1/i1) / (n2/i2) ))/sigma /int2;

	  }
	      
	}
	nrejected++;
      }
          
          
    }
  }

  result*=1000;

  return result;
};
