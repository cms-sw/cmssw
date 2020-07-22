"""
This service provides the logic of invalidating alru_cache.

If DQM GUI will ever be horizontally scaled, the functionality of invalidating caches 
needs to be orchestrated with the rest of the running instances!!!
"""

from functools import _make_key


class CacheInvalidationService:
    
    @classmethod
    def invalidate_on_new_sample(cls, service, sample):
        """
        This should be called whenever a new sample is registered.
        This function will loop over all sample related caches and invalidate the appropriate ones.

        service is the class object of the cached method.
        sample is of type FullSample and indicates the sample that was just registered.
        """

        cls.__invalidate_get_samples(service, sample)
        cls.__invalidate_get_archive(service, sample)
        cls.__invalidate_get_me_names_list(service, sample)
        cls.__invalidate_get_me_infos_list(service, sample)
        cls.__invalidate_get_filename_fileformat_names_infos(service, sample)


    @classmethod
    def __invalidate_get_samples(cls, service, sample):
        cache = service.get_samples._cache
        keys_to_remove = []
        dataset = sample.dataset.lower()

        for key in cache:
            if sample.run == key[1] and sample.lumi == key[3]:
                if key[2].lower() in dataset:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in cache:
                cache.pop(key)


    @classmethod
    def __invalidate_get_archive(cls, service, sample):
        cache = service.get_archive._cache
        keys_to_remove = []

        for key in cache:
            # We don't care about search and path arguments - wipe out caches 
            # for all search terms and pathes if other arguments match
            if sample.run == key[1] and sample.dataset == key[2] and sample.lumi == key[5]:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if key in cache:
                cache.pop(key)


    @classmethod
    def __invalidate_get_me_names_list(cls, service, sample):
        wrapped = service._GUIService__get_me_names_list
        wrapped.invalidate(*(service, sample.dataset, sample.run, sample.lumi))

    
    @classmethod
    def __invalidate_get_me_infos_list(cls, service, sample):
        wrapped = service._GUIService__get_me_infos_list
        wrapped.invalidate(*(service, sample.dataset, sample.run, sample.lumi))


    @classmethod
    def __invalidate_get_filename_fileformat_names_infos(cls, service, sample):
        wrapped = service._GUIService__get_filename_fileformat_names_infos
        wrapped.invalidate(*(service, sample.dataset, sample.run, sample.lumi))
