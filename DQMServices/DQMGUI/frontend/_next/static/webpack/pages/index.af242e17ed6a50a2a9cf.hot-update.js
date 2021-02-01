webpackHotUpdate_N_E("pages/index",{

/***/ "./hooks/useUpdateInLiveMode.tsx":
/*!***************************************!*\
  !*** ./hooks/useUpdateInLiveMode.tsx ***!
  \***************************************/
/*! exports provided: useUpdateLiveMode */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "useUpdateLiveMode", function() { return useUpdateLiveMode; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);


var _s = $RefreshSig$();




var useUpdateLiveMode = function useUpdateLiveMode() {
  _s();

  var current_time = new Date().getTime();

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](current_time),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      not_older_than = _React$useState2[0],
      set_not_older_than = _React$useState2[1];

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_1__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_2__["store"]),
      set_updated_by_not_older_than = _React$useContext.set_updated_by_not_older_than,
      update = _React$useContext.update,
      set_update = _React$useContext.set_update;

  console.log(current_time);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;

  var create_an_interval = function create_an_interval() {
    var interval = setInterval(function () {
      set_not_older_than(function () {
        // 1 sec is 1000 milisec. we dividing by 10000 and multiply by 20, because we need to
        // have rounded sec. for exmaple: if it is 13, we need to have 20, or 36, we need to have 20 and etc.
        var seconds = Math.round(new Date().getTime() / 1000) + 20 * 1000;
        return seconds;
      });
    }, 20000);
    return interval;
  };

  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    var interval = create_an_interval();

    if (!update) {
      clearInterval(interval);
    }
  }, [update, query.run_number, query.dataset_name, query.folder_path, query.search_dataset_name, query.search_run_number]);
  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    if (update) {
      set_updated_by_not_older_than(not_older_than);
    }
  }, [not_older_than, update]);
  return {
    not_older_than: not_older_than,
    set_update: set_update,
    update: update
  };
};

_s(useUpdateLiveMode, "H3YrUHqiotQtF7feXEi6KEexCu0=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"]];
});

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZS50c3giXSwibmFtZXMiOlsidXNlVXBkYXRlTGl2ZU1vZGUiLCJjdXJyZW50X3RpbWUiLCJEYXRlIiwiZ2V0VGltZSIsIlJlYWN0Iiwibm90X29sZGVyX3RoYW4iLCJzZXRfbm90X29sZGVyX3RoYW4iLCJzdG9yZSIsInNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIiwidXBkYXRlIiwic2V0X3VwZGF0ZSIsImNvbnNvbGUiLCJsb2ciLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsImNyZWF0ZV9hbl9pbnRlcnZhbCIsImludGVydmFsIiwic2V0SW50ZXJ2YWwiLCJzZWNvbmRzIiwiTWF0aCIsInJvdW5kIiwiY2xlYXJJbnRlcnZhbCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJmb2xkZXJfcGF0aCIsInNlYXJjaF9kYXRhc2V0X25hbWUiLCJzZWFyY2hfcnVuX251bWJlciJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFDQTtBQUdPLElBQU1BLGlCQUFpQixHQUFHLFNBQXBCQSxpQkFBb0IsR0FBTTtBQUFBOztBQUNyQyxNQUFNQyxZQUFZLEdBQUcsSUFBSUMsSUFBSixHQUFXQyxPQUFYLEVBQXJCOztBQURxQyx3QkFHUUMsOENBQUEsQ0FBZUgsWUFBZixDQUhSO0FBQUE7QUFBQSxNQUc5QkksY0FIOEI7QUFBQSxNQUdkQyxrQkFIYzs7QUFBQSwwQkFRakNGLGdEQUFBLENBQWlCRywrREFBakIsQ0FSaUM7QUFBQSxNQUtuQ0MsNkJBTG1DLHFCQUtuQ0EsNkJBTG1DO0FBQUEsTUFNbkNDLE1BTm1DLHFCQU1uQ0EsTUFObUM7QUFBQSxNQU9uQ0MsVUFQbUMscUJBT25DQSxVQVBtQzs7QUFTckNDLFNBQU8sQ0FBQ0MsR0FBUixDQUFZWCxZQUFaO0FBQ0EsTUFBTVksTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7O0FBRUEsTUFBTUMsa0JBQWtCLEdBQUcsU0FBckJBLGtCQUFxQixHQUFNO0FBQy9CLFFBQU1DLFFBQVEsR0FBR0MsV0FBVyxDQUFDLFlBQU07QUFDakNaLHdCQUFrQixDQUFDLFlBQU07QUFDdkI7QUFDQTtBQUNBLFlBQU1hLE9BQU8sR0FBR0MsSUFBSSxDQUFDQyxLQUFMLENBQVcsSUFBSW5CLElBQUosR0FBV0MsT0FBWCxLQUF1QixJQUFsQyxJQUEwQyxLQUFLLElBQS9EO0FBQ0EsZUFBT2dCLE9BQVA7QUFDRCxPQUxpQixDQUFsQjtBQU1ELEtBUDJCLEVBT3pCLEtBUHlCLENBQTVCO0FBUUEsV0FBT0YsUUFBUDtBQUNELEdBVkQ7O0FBWUFiLGlEQUFBLENBQWdCLFlBQU07QUFDcEIsUUFBTWEsUUFBUSxHQUFHRCxrQkFBa0IsRUFBbkM7O0FBQ0EsUUFBSSxDQUFDUCxNQUFMLEVBQWE7QUFDWGEsbUJBQWEsQ0FBQ0wsUUFBRCxDQUFiO0FBQ0Q7QUFDRixHQUxELEVBS0csQ0FDRFIsTUFEQyxFQUVETSxLQUFLLENBQUNRLFVBRkwsRUFHRFIsS0FBSyxDQUFDUyxZQUhMLEVBSURULEtBQUssQ0FBQ1UsV0FKTCxFQUtEVixLQUFLLENBQUNXLG1CQUxMLEVBTURYLEtBQUssQ0FBQ1ksaUJBTkwsQ0FMSDtBQWNBdkIsaURBQUEsQ0FBZ0IsWUFBTTtBQUNwQixRQUFJSyxNQUFKLEVBQVk7QUFDVkQsbUNBQTZCLENBQUNILGNBQUQsQ0FBN0I7QUFDRDtBQUNGLEdBSkQsRUFJRyxDQUFDQSxjQUFELEVBQWlCSSxNQUFqQixDQUpIO0FBTUEsU0FBTztBQUFFSixrQkFBYyxFQUFkQSxjQUFGO0FBQWtCSyxjQUFVLEVBQVZBLFVBQWxCO0FBQThCRCxVQUFNLEVBQU5BO0FBQTlCLEdBQVA7QUFDRCxDQTlDTTs7R0FBTVQsaUI7VUFVSWMscUQiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguYWYyNDJlMTdlZDZhNTBhMmE5Y2YuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuXHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5cclxuZXhwb3J0IGNvbnN0IHVzZVVwZGF0ZUxpdmVNb2RlID0gKCkgPT4ge1xyXG4gIGNvbnN0IGN1cnJlbnRfdGltZSA9IG5ldyBEYXRlKCkuZ2V0VGltZSgpO1xyXG5cclxuICBjb25zdCBbbm90X29sZGVyX3RoYW4sIHNldF9ub3Rfb2xkZXJfdGhhbl0gPSBSZWFjdC51c2VTdGF0ZShjdXJyZW50X3RpbWUpO1xyXG4gIGNvbnN0IHtcclxuICAgIHNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxyXG4gICAgdXBkYXRlLFxyXG4gICAgc2V0X3VwZGF0ZSxcclxuICB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XHJcbiAgY29uc29sZS5sb2coY3VycmVudF90aW1lKVxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG5cclxuICBjb25zdCBjcmVhdGVfYW5faW50ZXJ2YWwgPSAoKSA9PiB7XHJcbiAgICBjb25zdCBpbnRlcnZhbCA9IHNldEludGVydmFsKCgpID0+IHtcclxuICAgICAgc2V0X25vdF9vbGRlcl90aGFuKCgpID0+IHtcclxuICAgICAgICAvLyAxIHNlYyBpcyAxMDAwIG1pbGlzZWMuIHdlIGRpdmlkaW5nIGJ5IDEwMDAwIGFuZCBtdWx0aXBseSBieSAyMCwgYmVjYXVzZSB3ZSBuZWVkIHRvXHJcbiAgICAgICAgLy8gaGF2ZSByb3VuZGVkIHNlYy4gZm9yIGV4bWFwbGU6IGlmIGl0IGlzIDEzLCB3ZSBuZWVkIHRvIGhhdmUgMjAsIG9yIDM2LCB3ZSBuZWVkIHRvIGhhdmUgMjAgYW5kIGV0Yy5cclxuICAgICAgICBjb25zdCBzZWNvbmRzID0gTWF0aC5yb3VuZChuZXcgRGF0ZSgpLmdldFRpbWUoKSAvIDEwMDApICsgMjAgKiAxMDAwO1xyXG4gICAgICAgIHJldHVybiBzZWNvbmRzO1xyXG4gICAgICB9KTtcclxuICAgIH0sIDIwMDAwKTtcclxuICAgIHJldHVybiBpbnRlcnZhbDtcclxuICB9O1xyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgY29uc3QgaW50ZXJ2YWwgPSBjcmVhdGVfYW5faW50ZXJ2YWwoKTtcclxuICAgIGlmICghdXBkYXRlKSB7XHJcbiAgICAgIGNsZWFySW50ZXJ2YWwoaW50ZXJ2YWwpO1xyXG4gICAgfVxyXG4gIH0sIFtcclxuICAgIHVwZGF0ZSxcclxuICAgIHF1ZXJ5LnJ1bl9udW1iZXIsXHJcbiAgICBxdWVyeS5kYXRhc2V0X25hbWUsXHJcbiAgICBxdWVyeS5mb2xkZXJfcGF0aCxcclxuICAgIHF1ZXJ5LnNlYXJjaF9kYXRhc2V0X25hbWUsXHJcbiAgICBxdWVyeS5zZWFyY2hfcnVuX251bWJlcixcclxuICBdKTtcclxuXHJcbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGlmICh1cGRhdGUpIHtcclxuICAgICAgc2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4obm90X29sZGVyX3RoYW4pO1xyXG4gICAgfVxyXG4gIH0sIFtub3Rfb2xkZXJfdGhhbiwgdXBkYXRlXSk7XHJcblxyXG4gIHJldHVybiB7IG5vdF9vbGRlcl90aGFuLCBzZXRfdXBkYXRlLCB1cGRhdGUgfTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==