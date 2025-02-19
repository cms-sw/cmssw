{
  if (TClass::GetClass("edmtest::Thing")) {
    cout << "class already exists!" << endl;
    return 0;
  }
  cout << "class not present yet" << endl;

  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();
  if (!TClass::GetClass("edmtest::Thing")) {
    cout << "class still missing" << endl;
    return 1;
  }
  cout << "class loaded" << endl;
  return 0;
}
