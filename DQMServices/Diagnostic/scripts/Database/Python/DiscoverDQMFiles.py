import os

class DiscoverDQMFiles:

    def recursive_file_gen(self, mydir):
        for root, dirs, files in os.walk(mydir):
            for file in files:
                yield os.path.join(root, file)

    def filesList(self, sourceDir, type = ""):
        fullList = list(self.recursive_file_gen(sourceDir))
        reducedList = list()
        for file in fullList:
            if file.endswith(".root") and (file.find(type) != -1):
                # print "file passing", file
                reducedList.append(file)
        return reducedList
