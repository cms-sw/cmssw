#include<iostream>
#include<vector>
#include<string>

void printCharBits(unsigned char c){
 
        for (int i=7; i>= 0; --i) {
            int bit = ((c >> i) & 1);
            std::cout << " "<<bit;
        }
}

void packIntoString(std::vector<unsigned char> const& source,
                    vector<unsigned char>& package)
{   
unsigned int packInOneByte = 4;

unsigned int sizeOfPackage = 1+((source.size()-1)/packInOneByte); //Two bits per HLT
cout << "sizeOfPackage: "<<sizeOfPackage<< std::endl;

package.resize(sizeOfPackage);
memset(&package[0], 0x00, sizeOfPackage);

  for (unsigned int i = 0; i != source.size() ; ++i) {
      std::cout <<"i: "<<i<< std::endl;
      unsigned int whichByte = i/packInOneByte;   
      std::cout<<"whichByte: "<<whichByte<< std::endl;
      unsigned int indxWithinByte = i % packInOneByte;
      std::cout <<"indxWithinByte: "<<indxWithinByte<< std::endl;

      std::cout <<"source["<<i<<"] B4: ";
      printCharBits(source.at(i));cout<< std::endl;
      std::cout<<"Shiffted by "<<indxWithinByte*2<<" source["<<i<<"] ";
             printCharBits(source[i] << (indxWithinByte*2));cout<< std::endl;

      std::cout << "package["<<whichByte<<"] B4: ";
      printCharBits(package[whichByte]);cout<< std::endl;
      package[whichByte] = package[whichByte] | (source[i] << (indxWithinByte*2));
      std::cout << "package["<<whichByte<<"] After: ";
      printCharBits(package[whichByte]);cout<< std::endl;
      std::cout<<"\n\n\n************"<< std::endl;
  }
  std::cout<<"Packaged Bits"<< std::endl;
  for (unsigned int i=0; i !=package.size() ; ++i)
     printCharBits(package[i]);

}

int main()

{
//printCharBits(0xFF);
vector<unsigned char> source;

//Mind that Only 2 LSBs are Important
source.push_back(0);
source.push_back(1);
source.push_back(2);
source.push_back(3);
//source.push_back(5);

vector<unsigned char> hltbits;
packIntoString(source, hltbits);
cout<<"\nSource Was: \n";
string space="    ";
for (unsigned int i=source.size()-1;i!=-1;--i) {
     space=+"    ";
     std::cout<<space;
     printCharBits(source[i]);
}
     std::cout<< std::endl;
return 0;
}
