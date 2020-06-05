
from data_types import FileFormat
from reading.dqmclassic_reader import DQMCLASSICReader
from reading.dqmio_reader import DQMIOReader


class GUIMEReader:
    """
    This service reads monitor elements from files.
    Actual reading is delegated to file format specific readers.
    """

    @classmethod
    async def read(cls, filename, file_format, me_info):
        reader = cls.__pick_reader(file_format)
        return await reader.read(filename, me_info)


    @classmethod
    def __pick_reader(cls, file_format):
        """
        Picks the correct reader based on the file format that's being read.
        If a new file format are added, a reader has to be registered in this method.
        """

        if file_format == FileFormat.DQMCLASSIC:
            return DQMCLASSICReader()
        elif file_format == FileFormat.DQMIO:
            return DQMIOReader()
        return None
