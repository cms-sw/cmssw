#include <iostream>
#include "L1RCTNeighborMap.h"
using std::cout;
using std::endl;
void printVec(vector<int> vec){
  cout << "Elements are : ";  
  for(int i = 0; i<3; i++)
    cout << vec.at(i) << " ";
  cout << endl;
}
int main() {
  L1RCTNeighborMap nmap;
  for(int i=0; i<18; i++){
    for(int j=0; j<7; j++){
      for(int k=0;k<2; k++){
	cout << "North " << i << " " << j << " " << k << " "; 
	printVec(nmap.north(i,j,k));
	cout << "South " << i << " " << j << " " << k << " ";
	printVec(nmap.south(i,j,k));
	cout << "West " << i << " " << j << " " << k << " ";
	printVec(nmap.west(i,j,k));
	cout << "East " << i << " " << j << " " << k << " ";
	printVec(nmap.east(i,j,k));
      }
    }
  }
}
