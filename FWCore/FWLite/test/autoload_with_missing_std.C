{
//NOTE: this test causes the macro to abort since
// edmtest is unknown at this point
//if( TClass::GetClass("vector<edmtest::Thing>") ) {
//   cout <<"class already exists!"<<endl;
//   exit(1);
//}
//cout <<"class not present yet"<<endl;
gSystem->Load("libFWCoreFWLite");
AutoLibraryLoader::enable();
if( !TClass::GetClass("vector<edmtest::Thing>") ) {
   exit(1);
}
cout <<"class loaded"<<endl;
exit(0);
}
