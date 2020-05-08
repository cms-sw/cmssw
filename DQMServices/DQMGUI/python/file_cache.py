
class FileCache:
    __MAX_LENGTH = 20
    __cache = {}


    @classmethod
    def open(cls, file, mode='rb'):
        if file in cls.__cache:
            fo = cls.__cache[file]
            if not fo.closed:
                return fo


        fo = open(file, mode)
        cls.__cache[file] = fo

        return fo