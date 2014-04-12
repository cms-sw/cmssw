void psetDumpFromTTree(TString dir) {
    TTree *t = (TTree *) gFile->Get(dir+"/fitter_tree");
    TList *md = t->GetUserInfo();
    TObjString *obj;
    TIter next(md); 
    while ((obj = (TObjString*) next())) {
        std::cout << "   =========== PROCESS CONFIGURATION DUMP ===========   " << std::endl;
        std::cout << "   ==================================================   " << std::endl;
        std::cout << obj->GetString() << std::endl;
        std::cout << "   ==================================================   " << std::endl;
    }
}
