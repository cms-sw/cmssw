{
if( TClass::GetClass("edmtest::Thing") ) {
   cout <<"class already exists!"<<endl;
   exit(0);
}
cout <<"class not present yet"<<endl;

gSystem->Load("libFWCoreFWLite");
FWLiteEnabler::enable();
if( !TClass::GetClass("edmtest::Thing") ) {
   cout <<"class still missing"<<endl;
   exit(1);
}
cout <<"class loaded"<<endl;
exit(0);
}
