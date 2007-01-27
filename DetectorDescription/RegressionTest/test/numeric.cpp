#include<limits>
#include<iostream>
#include<vector>
#include<fstream>

using namespace std;

void print(double d)
{
  cout.precision(26);
  cout.setf(ios_base::scientific,ios_base::floatfield);
  cout << d << endl;
}

int main()
{
  numeric_limits<double> nl;
  cout << "radix   = " << nl.radix << endl
       << "digits  = " << nl.digits << endl
       << "digits10= " << nl.digits10 << endl
    ; 

  double d1(1), d10(10), dpi(3.141592);
  print(d1);
  print(d10);
  print(dpi);
  size_t sz(100000);
  vector<double> v(sz);
  vector<double>::iterator it(v.begin()), ed(v.end());
  size_t count(0);
  ofstream file("vector.txt");
  for(;it!=ed;++it) {
    *it = count++;
    double d(count);
    file.write((char *)(&d),sizeof(double));
  }
  
  
  
  cout << count << " non-zero entries." << endl; 
  return 0;
}
